"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a low-rank decomposition for 3x3 * 3x7 matrix multiplication.
    Uses block-Strassen for the 3x3x3 part (rank 26) and schoolbook for 3x3x4 (rank 36).
    Then attempts to reduce rank by finding redundant/mergeable terms.
    Falls back to schoolbook rank 63 if validation fails.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit

    def validate(U, V, W, tol=1e-9):
        """Check if the decomposition is exact using NumPy."""
        try:
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            n, m, p = DIMENSIONS
            rank = U_arr.shape[1]
            
            # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        s = np.dot(np.dot(U_arr[u_idx, :], V_arr[v_idx, :]), W_arr[w_idx, :])
                        if abs(s - 1.0) > tol:
                            return False
            return True
        except Exception:
            return False

    def construct_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def construct_rank_62():
        """
        Block-Strassen decomposition:
        - 3x3x3 part: rank 26 (Strassen for 2x2 block + schoolbook for cross terms)
        - 3x3x4 part: rank 36 (schoolbook)
        Total: 62
        """
        rank_3x3x3 = 26
        rank_3x3x4 = 36
        total_rank = rank_3x3x3 + rank_3x3x4
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        
        def add_component(u_coeffs, v_coeffs, w_coeffs):
            """Add one rank-1 component with given coefficient dicts."""
            nonlocal col
            for u_idx, coeff_u in u_coeffs.items():
                U[u_idx][col] += coeff_u
            for v_idx, coeff_v in v_coeffs.items():
                V[v_idx][col] += coeff_v
            for w_idx, coeff_w in w_coeffs.items():
                W[w_idx][col] += coeff_w
            col += 1

        # --- Part 1: 3x3x3 (k=0,1,2) ---
        # A indices: A[i,j] -> i*3+j: A00=0, A01=1, A02=2, A10=3, A11=4, A12=5, A20=6, A21=7, A22=8
        # B indices: B[j,k] -> j*3+k: B00=0, B01=1, B02=2, B10=3, B11=4, B12=5, B20=6, B21=7, B22=8
        # C indices: C[k,i] -> k*3+i: C00=0, C01=1, C02=2, C10=3, C11=4, C12=5, C20=6, C21=7, C22=8
        
        # Strassen for top-left 2x2 blocks: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
        add_component({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1})
        # P2 = (A10+A11)(B00) -> C10 += P2, C11 -= P2
        add_component({3:1, 4:1}, {0:1}, {3:1, 4:-1})
        # P3 = A00(B10-B11) -> C01 += P3, C11 += P3
        add_component({0:1}, {3:1, 4:-1}, {1:1, 4:1})
        # P4 = A11(B01-B00) -> C00 += P4, C10 += P4
        add_component({4:1}, {1:1, 0:-1}, {0:1, 3:1})
        # P5 = (A00+A01)(B11) -> C00 -= P5, C01 += P5
        add_component({0:1, 1:1}, {4:1}, {0:-1, 1:1})
        # P6 = (A10-A00)(B00+B01) -> C11 += P6
        add_component({3:1, 0:-1}, {0:1, 1:1}, {4:1})
        # P7 = (A01-A11)(B10+B11) -> C00 += P7
        add_component({1:1, 4:-1}, {3:1, 4:1}, {0:1})
        
        # Cross terms for 3x3x3:
        # C[0:2, 0:2] += A[0:2,2] * B[2,0:2]
        add_component({2:1}, {6:1}, {0:1})   # C00 += A02*B20
        add_component({2:1}, {7:1}, {1:1})   # C01 += A02*B21
        add_component({5:1}, {6:1}, {3:1})   # C10 += A12*B20
        add_component({5:1}, {7:1}, {4:1})   # C11 += A12*B21
        
        # C[2, 0:2] += A[0:2,0:2] * B[0:2,2]
        add_component({0:1}, {2:1}, {6:1})   # C20 += A00*B02
        add_component({3:1}, {5:1}, {6:1})   # C20 += A10*B12
        add_component({1:1}, {2:1}, {7:1})   # C21 += A01*B02
        add_component({4:1}, {5:1}, {7:1})   # C21 += A11*B12
        
        # C[0:2, 2] += A[2,0:2] * B[0:2,0:2]
        add_component({6:1}, {0:1}, {2:1})   # C02 += A20*B00
        add_component({7:1}, {3:1}, {2:1})   # C02 += A21*B10
        add_component({6:1}, {1:1}, {5:1})   # C12 += A20*B01
        add_component({7:1}, {4:1}, {5:1})   # C12 += A21*B11
        
        # C[2, 0:2] += A[0:2,2] * B[2,2]
        add_component({2:1}, {8:1}, {6:1})   # C20 += A02*B22
        add_component({5:1}, {8:1}, {7:1})   # C21 += A12*B22
        
        # C[2, 2] += A[2,0:2] * B[0:2,2]
        add_component({6:1}, {2:1}, {8:1})   # C22 += A20*B02
        add_component({7:1}, {5:1}, {8:1})   # C22 += A21*B12
        
        # C[2, 0:2] += A[2,2] * B[2,0:2]
        add_component({8:1}, {6:1}, {6:1})   # C20 += A22*B20
        add_component({8:1}, {7:1}, {7:1})   # C21 += A22*B21
        
        # C[2, 2] += A[2,2] * B[2,2]
        add_component({8:1}, {8:1}, {8:1})   # C22 += A22*B22
        
        # --- Part 2: 3x3x4 (k=3,4,5,6) Schoolbook ---
        for k in range(3, 7):
            for i in range(3):
                for j in range(3):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
                    
        return {"U": U, "V": V, "W": W}

    def try_reduce_rank(U, V, W, max_rank, time_limit):
        """
        Try to reduce rank by removing zero columns or merging equivalent terms.
        Uses a simple greedy approach: check if any column can be removed
        by expressing it as a combination of others.
        """
        start = time.time()
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank = U_arr.shape[1]
        
        # First, remove any zero columns
        nonzero_cols = []
        for c in range(rank):
            if np.any(U_arr[:, c] != 0) or np.any(V_arr[:, c] != 0) or np.any(W_arr[:, c] != 0):
                nonzero_cols.append(c)
        
        if len(nonzero_cols) < rank:
            U_arr = U_arr[:, nonzero_cols]
            V_arr = V_arr[:, nonzero_cols]
            W_arr = W_arr[:, nonzero_cols]
            rank = len(nonzero_cols)
        
        # Try to find linearly dependent columns
        # Build the tensor and check rank via SVD on unfolded tensors
        # For small ranks, try removing one column at a time and revalidating
        
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        best_rank = rank
        
        for c in range(rank):
            if time.time() - start > time_limit:
                break
            
            # Try removing column c
            mask = [i for i in range(rank) if i != c]
            U_test = U_arr[:, mask]
            V_test = V_arr[:, mask]
            W_test = W_arr[:, mask]
            
            if validate(U_test.tolist(), V_test.tolist(), W_test.tolist()):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank = len(mask)
        
        if best_rank < rank:
            return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank
        return None

    # Attempt rank 62
    best_result = None
    try:
        result_62 = construct_rank_62()
        if validate(result_62["U"], result_62["V"], result_62["W"]):
            best_result = result_62
            print(f"Rank 62 decomposition validated successfully.")
            
            # Try to reduce further
            time_remaining = deadline - time.time()
            if time_remaining > 5:
                reduced = try_reduce_rank(
                    result_62["U"], result_62["V"], result_62["W"],
                    max_rank=62, time_limit=min(time_remaining - 2, 60)
                )
                if reduced is not None:
                    U_red, V_red, W_red, new_rank = reduced
                    if validate(U_red, V_red, W_red):
                        best_result = {"U": U_red, "V": V_red, "W": W_red}
                        print(f"Reduced to rank {new_rank}.")
    except Exception as e:
        print(f"Rank 62 construction failed: {e}")
    
    # Fallback to schoolbook
    if best_result is None:
        print("Falling back to schoolbook rank 63.")
        best_result = construct_schoolbook()
    
    return best_result

# EVOLVE-BLOCK-END
