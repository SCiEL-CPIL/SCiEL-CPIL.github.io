    def local_search(target_rank, max_iters=50000):
        # Initialize randomly
        U = np.random.randint(-1, 2, (n*m, target_rank))
        V = np.random.randint(-1, 2, (m*p, target_rank))
        W = np.random.randint(-1, 2, (p*n, target_rank))
        
        # Precompute K_r = V[:, r] o W[:, r]
        # K shape: (R, m*p, p*n)
        K = np.einsum('br,cr->bcr', V, W, optimize=True)
        
        S = np.einsum('ar,bcr->abc', U, K, optimize=True)
        current_error = np.sum(np.abs(S - T))
        
        values = np.array([-1, 0, 1])
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick random matrix and index
            mat_choice = random.randint(0, 2)
            if mat_choice == 0:
                i = random.randint(0, n*m - 1)
                r = random.randint(0, target_rank - 1)
                old_val = U[i, r]
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                # Change in S is delta * K[r]
                new_S = S + delta * K[r]
                new_error = np.sum(np.abs(new_S - T))
                if new_error < current_error:
                    U[i, r] = new_val
                    S = new_S
                    current_error = new_error
            elif mat_choice == 1:
                j = random.randint(0, m*p - 1)
                r = random.randint(0, target_rank - 1)
                old_val = V[j, r]
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                # Change in S: S[:, j, :] += delta * outer(U[:, r], W[:, r])
                old_col = S[:, j, :]
                update_mat = delta * np.outer(U[:, r], W[:, r])
                new_col = old_col + update_mat
                old_err_part = np.sum(np.abs(old_col - T[:, j, :]))
                new_err_part = np.sum(np.abs(new_col - T[:, j, :]))
                new_error = current_error - old_err_part + new_err_part
                
                if new_error < current_error:
                    V[j, r] = new_val
                    S[:, j, :] = new_col
                    # Update K[r]
                    K[r, j, :] = new_val * W[:, r]
                    current_error = new_error
            else:
                k = random.randint(0, p*n - 1)
                r = random.randint(0, target_rank - 1)
                old_val = W[k, r]
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                # Change in S: S[:, :, k] += delta * outer(U[:, r], V[:, r])
                old_col = S[:, :, k]
                update_mat = delta * np.outer(U[:, r], V[:, r])
                new_col = old_col + update_mat
                old_err_part = np.sum(np.abs(old_col - T[:, :, k]))
                new_err_part = np.sum(np.abs(new_col - T[:, :, k]))
                new_error = current_error - old_err_part + new_err_part
                
                if new_error < current_error:
                    W[k, r] = new_val
                    S[:, :, k] = new_col
                    # Update K[r]
                    K[r, :, k] = V[:, r] * new_val
                    current_error = new_error

            if current_error == 0:
                return U, V, W
        
        return None
