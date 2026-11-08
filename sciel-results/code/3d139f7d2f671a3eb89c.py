    def verify(U, V, W):
        """Verify tensor decomposition exactly using numpy."""
        r = U.shape[1]
        if r == 0:
            return True
        
        # U: (n, m, r), V: (m, p, r), W: (p, n, r)
        # Note: W is indexed by (k, i) in the tensor sense, stored as k*n+i
        # So W.reshape(p, n, r) gives W[k, i, r]
        
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        # Construct the full tensor T of shape (n, m, m, p, p, n)
        # T[i, j, jp, k, kp, ip] = sum_r U3[i,j,r] * V3[jp,k,r] * W3[kp,ip,r]
        
        # We want to check if T equals the identity tensor
        # Identity tensor E[i, j, jp, k, kp, ip] = 1 if i==ip, j==jp, k==kp else 0
        
        # Use einsum for efficiency
        # T = np.einsum('ijr,jkr,kir->ijkik', U3, V3, W3) -- wait, indices
        # U3: i, j, r
        # V3: jp, k, r  (j' is jp)
        # W3: kp, ip, r (k' is kp, i' is ip)
        
        # Result should be indexed by i, j, jp, k, kp, ip
        # Let's compute T[i, j, jp, k, kp, ip]
        # This is sum_r U3[i, j, r] V3[jp, k, r] W3[kp, ip, r]
        
        T = np.einsum('ijr,jkr,kir->ijkik', U3, V3, W3) 
        # Wait, W3 is (p, n, r) -> indices kp, ip, r.
        # The einsum string 'kir' implies k, i, r. But W3 is k, i, r?
        # W3 shape is (p, n, r). Axis 0 is k (size p), Axis 1 is i (size n).
        # So W3[kp, ip, r].
        # The einsum 'kir' matches if we map k->kp, i->ip.
        
        # Let's construct the expected tensor
        # E[i, j, jp, k, kp, ip]
        # i: 0..n-1
        # j: 0..m-1
        # jp: 0..m-1
        # k: 0..p-1
        # kp: 0..p-1
        # ip: 0..n-1
        
        # This is a large tensor: 2 * 3 * 3 * 7 * 7 * 2 = 1764 elements.
        # We can create it by broadcasting.
        
        # Identity for i and ip:
        I_i = np.eye(n, dtype=int) # (n, n) -> I_i[i, ip]
        # Identity for j and jp:
        I_j = np.eye(m, dtype=int) # (m, m) -> I_j[j, jp]
        # Identity for k and kp:
        I_k = np.eye(p, dtype=int) # (p, p) -> I_k[k, kp]
        
        # Combine them
        # We need E[i, j, jp, k, kp, ip] = I_i[i, ip] * I_j[j, jp] * I_k[k, kp]
        
        # Reshape to broadcast
        # I_i[i, ip] -> (n, 1, 1, 1, 1, n)
        # I_j[j, jp] -> (1, m, m, 1, 1, 1)
        # I_k[k, kp] -> (1, 1, 1, p, p, 1)
        
        E = (I_i[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :] *
             I_j[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis] *
             I_k[np.newaxis, np.newaxis, np.newaxis, :, :, np.newaxis])
        
        # T shape from einsum:
        # 'ijr,jkr,kir->ijkik'
        # i: n, j: m, j (from V is jp? No, V is jkr. j is axis 0 of V? 
        # V3 is (m, p, r). So V3[jp, k, r].
        # In einsum 'jkr', j is axis 0 (size m), k is axis 1 (size p).
        # So V3 index is jp, k.
        # U3 is (n, m, r). U3[i, j, r].
        # W3 is (p, n, r). W3[kp, ip, r].
        # In einsum 'kir', k is axis 0 (size p), i is axis 1 (size n).
        # So W3 index is kp, ip.
        
        # Output indices: i, j, jp (from V's j), k (from V's k), kp (from W's k), ip (from W's i).
        # So output shape is (n, m, m, p, p, n).
        # This matches E shape (n, m, m, p, p, n).
        
        return np.allclose(T, E, atol=1e-9)
