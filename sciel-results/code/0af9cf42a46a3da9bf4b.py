"""Find rank-minimal exact tensor decomposition for 2x4 by 4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    start_time = time.time()
    deadline = 140  # Leave 10s margin for safety
    
    # Build the target multiplication tensor
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # a in [0, 8), b in [0, 28), c in [0, 14)
    T_tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_tensor[a, b, c] = 1

    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        rank = U.shape[1]
        T_approx = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_approx += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        return np.allclose(T_tensor, T_approx, atol=1e-6)

    def schoolbook():
        """Fallback: rank = n*m*p = 56."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, column] = 1
                    V[b, column] = 1
                    W[c, column] = 1
                    column += 1
        return U, V, W

    def construct_rank_50():
        """
        Construct a rank-50 decomposition.
        
        Strategy:
        The multiplication $C = AB$ where $A$ is $2 \times 4$ and $B$ is $4 \times 7$.
        $C$ is $2 \times 7$.
        $C_{i,k} = \sum_{j=0}^{3} A_{i,j} B_{j,k}$.
        
        We can view this as two independent $1 \times 4 \times 7$ multiplications for row 0 and row 1 of A.
        However, we can do better by using the structure of the tensor.
        
        A known technique for small matrices is to use the "splitting" method or specific low-rank algorithms.
        For $1 \times 4 \times 7$, the rank is at least $\max(4, 7, 28) = 28$? No, rank of $1 \times m \times p$ is $m \times p$ in schoolbook.
        Wait, the rank of the $1 \times m \times p$ tensor is $m \times p$?
        $T_{j,k} = A_j B_k$. This is a rank-1 tensor in the matrix sense, but as a 3-tensor?
        Actually, the $1 \times m \times p$ multiplication tensor is $\sum_{j,k} e_j \otimes e_k \otimes e_{jk}$.
        This tensor has rank $m \times p$.
        
        So for $2 \times 4 \times 7$, if we treat it as two independent $1 \times 4 \times 7$ tensors, the rank is $2 \times 4 \times 7 = 56$.
        
        To get rank 50, we need to exploit that the "middle" dimension $m=4$ is shared.
        Let's look at the tensor $T_{i,j,k} = A_{i,j} B_{j,k}$.
        
        Consider the decomposition of the $4 \times 4$ multiplication tensor.
        The rank of $4 \times 4 \times 4$ is known to be 15 (or 16? No, 15 is the best known for 4x4x4? 
        Actually, the rank of $4 \times 4 \times 4$ is 15? No, it's 16? 
        Let's check: $4 \times 4 \times 4$ rank is 15? No, 16 is schoolbook. 
        Best known for $4 \times 4 \times 4$ is 15? 
        Wait, $2 \times 2 \times 2$ is 7.
        $4 \times 4 \times 4$ can be done with 15? 
        Actually, the rank of $4 \times 4 \times 4$ is 15? 
        Let's assume we can decompose the $4 \times 4 \times 4$ core.
        
        Alternative approach:
        The task mentions "rank-50 Strassen-based decomposition".
        Let's try to construct a rank-50 solution directly.
        
        We can write $C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
        
        Let's use the fact that $4 = 2 + 2$.
        Split $j$ into two groups of 2?
        
        Actually, there is a simpler constructive lower bound.
        The rank of $M_{2,4,7}$ is at most $2 \times 4 \times 7 - \text{something}$.
        
        Let's try to find a rank-50 solution by modifying the schoolbook solution.
        In schoolbook, we have 56 terms.
        Can we merge some?
        
        Let's look at the tensor entries:
        $T_{a,b,c}$.
        $a = 2j + i$ (since $n=2$, $m=4$, $a = i \cdot 4 + j$? No, $a = i \cdot m + j = 4i + j$).
        $b = j \cdot p + k = 7j + k$.
        $c = k \cdot n + i = 2k + i$.
        
        We want to represent $\sum_{i,j,k} A_{4i+j} B_{7j+k} C_{2k+i}$.
        
        Let's try a specific construction for rank 50.
        We can use the rank-15 decomposition of $4 \times 4 \times 4$ if we embed it.
        But $p=7$.
        
        Let's try a different structural approach:
        Use the rank-15 decomposition for $4 \times 4 \times 4$ for the first 4 columns of B, and rank-3 for the remaining 3?
        Rank of $1 \times 4 \times 3$ is 12? No.
        
        Let's go back to the "cell findings" hint: "verified rank-50 Strassen-based decomposition".
        This suggests a specific known result or a constructive method that yields 50.
        
        One such method:
        $M_{2,4,7} \le M_{2,4,4} \otimes M_{1,1,3}$? No.
        
        Let's try to compute the rank of $M_{2,4,7}$ using a simple heuristic:
        Rank $\le 2 \times 4 \times 7 - 6 = 50$.
        
        How to get 50?
        We can use the following identity:
        $\sum_{j=0}^3 A_{i,j} B_{j,k} = \sum_{j=0}^3 (A_{i,j} + c) (B_{j,k} + d) - \dots$
        This is the Winograd-style "adding and subtracting" to reduce terms.
        
        For $1 \times 2 \times 2$, rank is 4 (schoolbook) or 4? No, $1 \times 2 \times 2$ is rank 4?
        $C_k = \sum_j A_j B_{j,k}$.
        $C_0 = A_0 B_0 + A_1 B_1$.
        $C_1 = A_0 B_2 + A_1 B_3$.
        Rank 4.
        
        For $1 \times 3 \times 3$, rank is 9?
        
        Let's try to implement a generic "greedy" rank reduction on the schoolbook tensor.
        Start with rank 56.
        Try to find linear dependencies among the 56 rank-1 tensors.
        The 56 tensors are $e_a \otimes e_b \otimes e_c$.
        They are linearly independent in the space of tensors?
        No, the space of tensors is $8 \times 28 \times 14$.
        The 56 vectors are columns of a matrix if we flatten.
        But they are rank-1 tensors.
        
        If we form the matrix $M$ of size $(8 \cdot 28) \times 56$ where each column is the flattened tensor,
        if the columns are linearly independent, the rank is 56.
        Let's check if they are independent.
        $T_{i,j,k}$ corresponds to basis element $e_{4i+j} \otimes e_{7j+k} \otimes e_{2k+i}$.
        These are distinct basis elements in the tensor product space?
        No, the tensor product space basis is $e_a \otimes e_b \otimes e_c$.
        Each term in the sum is a single basis element?
        $T = \sum_{i,j,k} e_{4i+j} \otimes e_{7j+k} \otimes e_{2k+i}$.
        Since each $(i,j,k)$ gives a unique triple $(a,b,c)$, and the basis elements are linearly independent,
        the rank of this tensor is exactly the number of terms, which is 56.
        
        Wait, is that true?
        Yes, the "standard" or "schoolbook" decomposition uses distinct basis vectors for each term.
        Therefore, the rank is at most 56.
        Is it possible to have a lower rank?
        Yes, if we use non-basis vectors.
        
        So the rank is the minimum number of rank-1 tensors that sum to $T$.
        
        The "cell findings" say a rank-50 solution exists.
        How to find it?
        
        We can use the "Landsberg" or "Strassen" style algebraic manipulations.
        
        Let's try a specific known decomposition for $2 \times 4 \times 7$.
        Actually, there is a known result: $M_{2,4,7} \le 50$.
        
        Let's try to construct it by splitting $p=7$ into $4+3$.
        $C_{i,k} = \sum_{j=0}^3 A_{i,j} B_{j,k}$.
        Split $k$ into $k_1 \in \{0..3\}$ and $k_2 \in \{4..6\}$.
        
        Part 1: $k \in 0..3$.
        $C_{i,k} = \sum_{j=0}^3 A_{i,j} B_{j,k}$.
        This is $2 \times 4 \times 4$.
        
        Part 2: $k \in 4..6$.
        $C_{i,k} = \sum_{j=0}^3 A_{i,j} B_{j,k}$.
        This is $2 \times 4 \times 3$.
        
        Rank of $2 \times 4 \times 4$:
        We can use the rank-15 decomposition of $4 \times 4 \times 4$?
        No, $2 \times 4 \times 4$ is different.
        
        Let's use the fact that $2 \times 4 \times 4$ can be done with rank 16?
        Schoolbook is $2 \times 4 \times 4 = 32$.
        
        If we can do $2 \times 4 \times 4$ with rank 25 and $2 \times 4 \times 3$ with rank 25, total 50.
        
        Let's try to find a rank-25 decomposition for $2 \times 4 \times 4$.
        
        Actually, let's just try to run a solver for rank 50.
        But ALS is slow.
        
        Alternative:
        The problem asks for "integer/half-integer entries".
        This suggests a constructive algebraic solution, not a numerical one.
        
        Let's look at the $1 \times 2 \times 2$ case.
        $C_0 = A_0 B_0 + A_1 B_1$.
        $C_1 = A_0 B_2 + A_1 B_3$.
        Rank 4.
        
        $1 \times 3 \times 3$:
        $C_0 = A_0 B_0 + A_1 B_1 + A_2 B_2$.
        ...
        Rank 9.
        
        It seems for $1 \times m \times p$, the rank is $m \times p$.
        
        So for $2 \times 4 \times 7$, if we treat rows independently, rank is $2 \times 28 = 56$.
        
        To get 50, we must share factors between the two rows.
        
        Let $A = [A_0, A_1]$ where $A_i$ is a row vector.
        $C_0 = A_0 B$.
        $C_1 = A_1 B$.
        
        Consider the vector space of pairs $(A_0, A_1)$.
        
        Let's try the "Strassen" trick on the $2 \times 2$ block of the output?
        
        Given the time constraint and the hint about rank 50, I will implement a function that attempts to find a rank-50 solution using a specific algebraic construction if possible, or falls back to a verified rank-50 solution if I can hardcode one.
        
        Since I cannot reliably derive the rank-50 construction from scratch in 150s, I will use the "cell findings" hint that a rank-50 solution was found by another program.
        I will implement a search that specifically targets rank 50 using a more efficient solver (like using `scipy.optimize` or a better ALS initialization) or I will just return the schoolbook rank 56 if I can't find 50 quickly.
        
        However, the goal is to beat 1.0. Rank 50 gives score $56/50 = 1.12$.
        
        Let's try a simple heuristic:
        Try to merge pairs of schoolbook terms.
        Term $(i,j,k)$ and $(i', j', k')$.
        If we can write $e_a \otimes e_b \otimes e_c + e_{a'} \otimes e_{b'} \otimes e_{c'}$ as a sum of 1 rank-1 tensor? No.
        
        Let's try ALS for rank 50 with a good initialization.
        Initialize U, V, W with random values, but constrained to be sparse or structured?
        
        Actually, let's just try ALS for rank 50. It might converge.
        """
        # Try ALS for rank 50
        pass

    def als_decomposition(r, seed, max_iters=100):
        """Alternating Least Squares for tensor decomposition."""
        rng = np.random.default_rng(seed)
        U = rng.standard_normal((n*m, r)) * 0.1
        V = rng.standard_normal((m*p, r)) * 0.1
        W = rng.standard_normal((p*n, r)) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            B_u = np.zeros((m*p * p*n, r))
            for r_idx in range(r):
                B_u[:, r_idx] = np.kron(V[:, r_idx], W[:, r_idx])
            T_u = T_tensor.reshape(n*m, -1)
            try:
                U_new = T_u @ np.linalg.pinv(B_u.T)
            except Exception:
                return None
            
            # Update V
            B_v = np.zeros((n*m * p*n, r))
            for r_idx in range(r):
                B_v[:, r_idx] = np.kron(U[:, r_idx], W[:, r_idx])
            T_v = T_tensor.transpose(1, 0, 2).reshape(m*p, -1)
            try:
                V_new = T_v @ np.linalg.pinv(B_v.T)
            except Exception:
                return None
            
            # Update W
            B_w = np.zeros((n*m * m*p, r))
            for r_idx in range(r):
                B_w[:, r_idx] = np.kron(U[:, r_idx], V[:, r_idx])
            T_w = T_tensor.transpose(2, 0, 1).reshape(p*n, -1)
            try:
                W_new = T_w @ np.linalg.pinv(B_w.T)
            except Exception:
                return None
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            T_approx = np.zeros((n*m, m*p, p*n))
            for r_idx in range(r):
                T_approx += np.outer(np.outer(U[:, r_idx], V[:, r_idx]), W[:, r_idx])
            if np.allclose(T_tensor, T_approx, atol=1e-5):
                return U, V, W
        
        return None

    # Try rank 50 first
    for seed in [42, 123, 7, 99, 2023, 555]:
        if time.time() > deadline:
            break
        result = als_decomposition(50, seed, max_iters=50)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                # Round to half-integers
                def round_clean(M):
                    M_r = np.round(M * 2) / 2
                    if np.allclose(M, M_r, atol=1e-4):
                        return M_r
                    return M
                U = round_clean(U)
                V = round_clean(V)
                W = round_clean(W)
                if verify(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
