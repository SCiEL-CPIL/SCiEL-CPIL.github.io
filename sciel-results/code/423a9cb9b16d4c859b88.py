            U_grad = np.zeros_like(U_arr)
            V_grad = np.zeros_like(V_arr)
            W_grad = np.zeros_like(W_arr)
            
            # Compute contributions
            # contrib_U[i, r] = err[i] * V_arr[tensor_b[i], r] * W_arr[tensor_c[i], r]
            # We can compute the full (N, R) arrays and then scatter column by column.
            
            # To avoid large temporary arrays if possible, but N=36, R=32 is small.
            # Let's just do the scatter column by column.
            
            # Pre-compute V and W indexed arrays
            V_idx = V_arr[tensor_b, :] # (N, R)
            W_idx = W_arr[tensor_c, :] # (N, R)
            U_idx = U_arr[tensor_a, :] # (N, R)
            
            for r in range(R):
                # U_grad[a_i, r] += err[i] * V_idx[i, r] * W_idx[i, r]
                contrib_U_r = err * V_idx[:, r] * W_idx[:, r]
                np.add.at(U_grad[:, r], tensor_a, contrib_U_r)
                
                # V_grad[b_i, r] += err[i] * U_idx[i, r] * W_idx[i, r]
                contrib_V_r = err * U_idx[:, r] * W_idx[:, r]
                np.add.at(V_grad[:, r], tensor_b, contrib_V_r)
                
                # W_grad[c_i, r] += err[i] * U_idx[i, r] * V_idx[i, r]
                contrib_W_r = err * U_idx[:, r] * V_idx[:, r]
                np.add.at(W_grad[:, r], tensor_c, contrib_W_r)
