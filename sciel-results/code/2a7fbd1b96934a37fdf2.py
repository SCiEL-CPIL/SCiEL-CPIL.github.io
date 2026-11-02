    def recursive_decomp(n, m, p):
        # Base cases
        if n == 2 and m == 2 and p == 2:
            return strassen_2x2x2()
        
        if n == 1 or m == 1 or p == 1:
            return schoolbook(n, m, p)
            
        # If any dimension is > 2, split it
        # We prefer to split into 2 and rest
        # We need to decide which dimension to split.
        # Heuristic: Split the largest dimension? Or the one that allows 2x2x2 blocks?
        # Let's try splitting n, then m, then p if > 2.
        
        U_list = []
        V_list = []
        W_list = []
        
        # Split n
        if n > 2:
            n1, n2 = 2, n - 2
            # Sub-blocks: (n1, m, p) and (n2, m, p)
            # Note: The "blocks" here are contiguous in the index space.
            # A is n x m. Split rows 0..1 and 2..n-1.
            
            U1, V1, W1 = recursive_decomp(n1, m, p)
            U2, V2, W2 = recursive_decomp(n2, m, p)
            
            # Expand to full n x m x p
            # U1 is n1 x m x R1. U2 is n2 x m x R2.
            # Global U is n x m x (R1+R2)
            
            R1 = U1.shape[1]
            R2 = U2.shape[1]
            
            U_full = np.zeros((n*m, R1+R2))
            V_full = np.zeros((m*p, R1+R2))
            W_full = np.zeros((p*n, R1+R2))
            
            # Place U1
            # U1 indices: i in 0..n1-1, j in 0..m-1 -> i*m + j
            for i in range(n1):
                for j in range(m):
                    U_full[i*m+j, :R1] = U1[i*m+j, :]
            
            # Place U2
            # U2 indices: i in 0..n2-1, j in 0..m-1 -> (i+2)*m + j in global?
            # Wait, local index i in U2 corresponds to global row i + n1.
            for i in range(n2):
                for j in range(m):
                    U_full[(i+n1)*m+j, R1:] = U2[i*m+j, :]
            
            # V is same for both (since m, p unchanged)
            V_full[:, :R1] = V1
            V_full[:, R1:] = V2
            
            # W
            # W1 is p x n1 x R1. Indices k*n1 + i.
            # Global W is p x n x (R1+R2). Indices k*n + i.
            for k in range(p):
                for i in range(n1):
                    W_full[k*n+i, :R1] = W1[k*n1+i, :]
            
            for k in range(p):
                for i in range(n2):
                    W_full[k*n + (i+n1), R1:] = W2[k*n2+i, :]
            
            return U_full, V_full, W_full
            
        # Split m
        if m > 2:
            m1, m2 = 2, m - 2
            U1, V1, W1 = recursive_decomp(n, m1, p)
            U2, V2, W2 = recursive_decomp(n, m2, p)
            
            R1 = U1.shape[1]
            R2 = U2.shape[1]
            
            U_full = np.zeros((n*m, R1+R2))
            V_full = np.zeros((m*p, R1+R2))
            W_full = np.zeros((p*n, R1+R2))
            
            # U
            # U1: i in 0..n-1, j in 0..m1-1 -> i*m + j? No, local m1.
            # Local U1 index: i*m1 + j.
            # Global U index: i*m + j.
            for i in range(n):
                for j in range(m1):
                    U_full[i*m+j, :R1] = U1[i*m1+j, :]
            
            for i in range(n):
                for j in range(m2):
                    U_full[i*m + (j+m1), R1:] = U2[i*m2+j, :]
            
            # V
            # V1: j in 0..m1-1, k in 0..p-1 -> j*p + k.
            # Global V: j in 0..m-1, k in 0..p-1 -> j*p + k.
            for j in range(m1):
                for k in range(p):
                    V_full[j*p+k, :R1] = V1[j*p+k, :]
            
            for j in range(m2):
                for k in range(p):
                    V_full[(j+m1)*p+k, R1:] = V2[j*p+k, :]
            
            # W is same for both (n, p unchanged)
            W_full[:, :R1] = W1
            W_full[:, R1:] = W2
            
            return U_full, V_full, W_full
            
        # Split p
        if p > 2:
            p1, p2 = 2, p - 2
            U1, V1, W1 = recursive_decomp(n, m, p1)
            U2, V2, W2 = recursive_decomp(n, m, p2)
            
            R1 = U1.shape[1]
            R2 = U2.shape[1]
            
            U_full = np.zeros((n*m, R1+R2))
            V_full = np.zeros((m*p, R1+R2))
            W_full = np.zeros((p*n, R1+R2))
            
            # U is same
            U_full[:, :R1] = U1
            U_full[:, R1:] = U2
            
            # V
            # V1: j in 0..m-1, k in 0..p1-1 -> j*p1 + k.
            # Global V: j in 0..m-1, k in 0..p-1 -> j*p + k.
            for j in range(m):
                for k in range(p1):
                    V_full[j*p+k, :R1] = V1[j*p1+k, :]
            
            for j in range(m):
                for k in range(p2):
                    V_full[j*p + (k+p1), R1:] = V2[j*p2+k, :]
            
            # W
            # W1: k in 0..p1-1, i in 0..n-1 -> k*n + i.
            # Global W: k in 0..p-1, i in 0..n-1 -> k*n + i.
            for k in range(p1):
                for i in range(n):
                    W_full[k*n+i, :R1] = W1[k*n+i, :]
            
            for k in range(p2):
                for i in range(n):
                    W_full[(k+p1)*n+i, R1:] = W2[k*n+i, :]
            
            return U_full, V_full, W_full
            
        # Fallback (should not reach here if logic is correct)
        return schoolbook(n, m, p)
    