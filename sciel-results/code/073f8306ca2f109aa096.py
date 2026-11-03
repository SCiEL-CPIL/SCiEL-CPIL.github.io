import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_np[i * m + j] * V_np[j * p + k] * W_np[k * n + i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2_decomp():
        """
        For 2x2x2, Strassen gives rank 7.
        Let A be 2x2, B be 2x2, C = A*B be 2x2.
        Strassen formulas:
        M1 = (A00+A11)*(B00+B11)
        M2 = (A10+A11)*B00
        M3 = A00*(B01-B11)
        M4 = A11*(B10-B00)
        M5 = (A00+A01)*B11
        M6 = (A10-A00)*(B00+B01)
        M7 = (A01-A11)*(B10+B11)
        C00 = M1+M4-M5+M7
        C01 = M3+M5
        C10 = M2+M4
        C11 = M1-M2+M3+M6
        """
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)

        # Strassen terms
        # M1: A00+A11 times B00+B11
        U[0, 0] = 1; U[3, 0] = 1  # A00, A11
        V[0, 0] = 1; V[3, 0] = 1  # B00, B11
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = -1  # C00+=, C01-=, C10-=, C11+=
        # Actually W should map to C00, C01, C10, C11
        # Let me be more careful.
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # For 2x2x2: n=2, m=2, p=2
        # U[0,0]=A00, U[0,1]=A01, U[1,0]=A10, U[1,1]=A11
        # V[0,0]=B00, V[0,1]=B01, V[1,0]=B10, V[1,1]=B11
        # W[0,0]=C00, W[0,1]=C01, W[1,0]=C10, W[1,1]=C11

        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)

        # M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = -1  # C00 += M1, C11 -= M1? No, C11 += M1
        # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
        W[0, 0] = 1; W[3, 0] = 1  # C00 += M1, C11 += M1

        # M2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1  # C10 += M2, C11 -= M2

        # M3 = A00*(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1  # C01 += M3, C11 += M3

        # M4 = A11*(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1  # C00 += M4, C10 += M4

        # M5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1  # C00 -= M5, C01 += M5

        # M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # C11 += M6

        # M7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1  # C00 += M7

        return U.tolist(), V.tolist(), W.tolist()

    def try_strassen_blocks():
        """
        Try to construct a decomposition by partitioning into 2x2x2 blocks
        and using Strassen for each block, plus schoolbook for remainders.
        n=2, m=4, p=7
        We can think of A as 2x4, B as 4x7.
        Partition m into 2+2, p into 2+2+2+1, n into 2.
        Block (i-block=0, j-block, k-block) where i-block in {0}, j-block in {0,1}, k-block in {0,1,2,3}
        For k-block=3 (size 1), we use schoolbook.
        For k-blocks 0,1,2 (size 2), we can use Strassen if the j-block is size 2.
        j-blocks 0,1 are size 2.
        So blocks: (0,0,0), (0,0,1), (0,0,2) use Strassen (rank 7 each)
        (0,1,0), (0,1,1), (0,1,2) use Strassen (rank 7 each)
        (0,0,3), (0,1,3) use schoolbook (rank 2*1=2 each)
        Total: 6*7 + 2*2 = 42 + 4 = 46
        """
        # This is getting complex. Let me just try to find a rank 42 solution.
        # Actually, let me just do random search with a better initialization.
        pass

    def random_search_with_annealing(rank, time_limit):
        """
        Random search with simulated annealing.
        """
        U = np.random.choice([-2, -1, 0, 1, 2], size=(nm, rank))
        V = np.random.choice([-2, -1, 0, 1, 2], size=(mp, rank))
        W = np.random.choice([-2, -1, 0, 1, 2], size=(pn, rank))

        def compute_error(U, V, W):
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                        err += (val - 1.0) ** 2
            return err

        current_err = compute_error(U, V, W)

        if current_err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()

        temperature = 10.0
        cooling = 0.99

        while time.time() < time_limit and temperature > 0.01:
            # Pick random entry
            mat = np.random.randint(0, 3)
            if mat == 0:
                idx = np.random.randint(0, nm)
                r = np.random.randint(0, rank)
                old = U[idx, r]
                new = old + np.random.choice([-2, -1, 1, 2])
                if new < -4 or new > 4:
                    continue
                U[idx, r] = new
                new_err = compute_error(U, V, W)
                delta = new_err - current_err
                if delta < 0 or np.random.random() < np.exp(-delta / temperature):
                    current_err = new_err
                else:
                    U[idx, r] = old
            elif mat == 1:
                idx = np.random.randint(0, mp)
                r = np.random.randint(0, rank)
                old = V[idx, r]
                new = old + np.random.choice([-2, -1, 1, 2])
                if new < -4 or new > 4:
                    continue
                V[idx, r] = new
                new_err = compute_error(U, V, W)
                delta = new_err - current_err
                if delta < 0 or np.random.random() < np.exp(-delta / temperature):
                    current_err = new_err
                else:
                    V[idx, r] = old
            else:
                idx = np.random.randint(0, pn)
                r = np.random.randint(0, rank)
                old = W[idx, r]
                new = old + np.random.choice([-2, -1, 1, 2])
                if new < -4 or new > 4:
                    continue
                W[idx, r] = new
                new_err = compute_error(U, V, W)
                delta = new_err - current_err
                if delta < 0 or np.random.random() < np.exp(-delta / temperature):
                    current_err = new_err
                else:
                    W[idx, r] = old

            temperature *= cooling

            if current_err < 1e-9:
                return U.tolist(), V.tolist(), W.tolist()

        if current_err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
        return None

    def greedy_construct(rank):
        """
        Greedy construction: for each rank position, choose U, V, W entries
        to best reduce the remaining error.
        """
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)

        # Target tensor
        T = np.ones((n, m, p))

        def current_decomp(U, V, W):
            result = np.zeros((n, m, p))
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            result[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
            return result

        for r in range(rank):
            if time.time() > deadline - 30.0:
                break

            # Try to find U[:, r], V[:, r], W[:, r] that minimize error
            # This is a 3-way factor, so we can use a simple iterative approach
            # Initialize
            U[:, r] = np.random.choice([-1, 0, 1], size=nm)
            V[:, r] = np.random.choice([-1, 0, 1], size=mp)
            W[:, r] = np.random.choice([-1, 0, 1], size=pn)

            for _ in range(100):
                if time.time() > deadline - 30.0:
                    break

                # Update U
                best_U = U[:, r].copy()
                for i in range(nm):
                    old = U[i, r]
                    for new_val in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
                        U[i, r] = new_val
                        decomp = current_decomp(U, V, W)
                        err = np.sum((decomp - T) ** 2)
                        if err < np.sum((current_decomp(best_U.reshape(-1, 1), V, W) - T) ** 2):
                            best_U[i] = new_val
                    U[i, r] = best_U[i]
                U[:, r] = best_U

                # Update V
                best_V = V[:, r].copy()
                for j in range(mp):
                    old = V[j, r]
                    for new_val in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
                        V[j, r] = new_val
                        decomp = current_decomp(U, V, W)
                        err = np.sum((decomp - T) ** 2)
                        if err < np.sum((current_decomp(U, best_V.reshape(-1, 1), W) - T) ** 2):
                            best_V[j] = new_val
                    V[j, r] = best_V[j]
                V[:, r] = best_V

                # Update W
                best_W = W[:, r].copy()
                for k in range(pn):
                    old = W[k, r]
                    for new_val in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
                        W[k, r] = new_val
                        decomp = current_decomp(U, V, W)
                        err = np.sum((decomp - T) ** 2)
                        if err < np.sum((current_decomp(U, V, best_W.reshape(-1, 1)) - T) ** 2):
                            best_W[k] = new_val
                    W[k, r] = best_W[k]
                W[:, r] = best_W

            decomp = current_decomp(U, V, W)
            if np.max(np.abs(decomp - T)) < 1e-9:
                return U.tolist(), V.tolist(), W.tolist()

        return None

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p

    # Try random search with annealing for various ranks
    for rank in [32, 34, 36, 38, 40, 42, 44, 46]:
        if time.time() > deadline - 40.0:
            break

        for attempt in range(10):
            if time.time() > deadline - 40.0:
                break

            result = random_search_with_annealing(rank, time.time() + 5.0)
            if result is not None:
                U, V, W = result
                if verify(U, V, W):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    break

        if best_rank == rank:
            break

    return best
