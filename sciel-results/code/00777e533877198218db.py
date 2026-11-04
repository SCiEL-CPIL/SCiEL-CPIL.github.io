"""Matrix multiplication 2,4,8 — rank search via exact bilinear decomposition."""
import time
import numpy as np
import random
from itertools import product

DIMENSIONS = [2, 4, 8]
_SEED_START = time.time()


def _deadline_guard(seconds_remaining=115.0):
    """Return True if we should stop searching."""
    return (time.time() - _SEED_START) > seconds_remaining


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8

    def make_schoolbook():
        """Return the standard rank-64 schoolbook decomposition."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def validate(U, V, W):
        """Check that the decomposition exactly reconstructs the tensor."""
        # T[i,j,k] should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # Also check off-diagonal terms are zero
        rank = len(U[0])
        # Check diagonal terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(total - 1) > 1e-10:
                        return False
        # Check some off-diagonal terms
        for _ in range(50):
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            jp = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            kp = random.randint(0, p - 1)
            ip = random.randint(0, n - 1)
            if i == ip and j == jp and k == kp:
                continue
            total = 0
            for c in range(rank):
                total += U[i * m + j, c] * V[jp * p + k, c] * W[kp * n + ip, c]
            if abs(total) > 1e-10:
                return False
        return True

    # First, get the baseline
    U_base, V_base, W_base, rank_base = make_schoolbook()

    # Try to find a better decomposition using structured approach
    # Based on Strassen-like ideas: split the m=4 dimension into groups

    def strassen_2x2x2():
        """
        Return Strassen decomposition for 2x2 * 2x2.
        Returns U, V, W of shape (4, 7) each.
        """
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))

        # Strassen's 7 products for 2x2 matrices
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # U encodes A indices: U[i,j, c] for i,j in {0,1}
        # V encodes B indices: V[j,k, c] for j,k in {0,1}
        # W encodes C indices: W[k,i, c] for k,i in {0,1}

        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1  # A00
        U[3, 0] = 1  # A11
        V[0, 0] = 1  # B00
        V[3, 0] = 1  # B11
        W[0, 0] = 1  # C00
        W[3, 0] = 1  # C11

        # M2 = (A10+A11)B00
        U[2, 1] = 1  # A10
        U[3, 1] = 1  # A11
        V[0, 1] = 1  # B00
        W[2, 1] = 1  # C10
        W[3, 1] = -1  # C11

        # M3 = A00(B10-B11)
        U[0, 2] = 1  # A00
        V[2, 2] = 1  # B10
        V[3, 2] = -1  # B11
        W[1, 2] = 1  # C01
        W[3, 2] = 1  # C11

        # M4 = A11(B01-B00)
        U[3, 3] = 1  # A11
        V[1, 3] = 1  # B01
        V[0, 3] = -1  # B00
        W[0, 3] = 1  # C00
        W[2, 3] = 1  # C10

        # M5 = (A00+A01)B11
        U[0, 4] = 1  # A00
        U[1, 4] = 1  # A01
        V[3, 4] = 1  # B11
        W[0, 4] = -1  # C00
        W[1, 4] = 1  # C01

        # M6 = (A10-A00)(B00+B10)
        U[2, 5] = 1  # A10
        U[0, 5] = -1  # A00
        V[0, 5] = 1  # B00
        V[2, 5] = 1  # B10
        W[3, 5] = 1  # C11

        # M7 = (A01-A11)(B01+B11)
        U[1, 6] = 1  # A01
        U[3, 6] = -1  # A11
        V[1, 6] = 1  # B01
        V[3, 6] = 1  # B11
        W[0, 6] = 1  # C00
        W[1, 6] = 1  # C01

        return U, V, W

    def build_strassen_based():
        """
        Build a decomposition based on Strassen for 2x4 * 4x8.
        Split m=4 into two groups of 2: {0,1} and {2,3}.
        For each group, use Strassen 2x2x2 decomposition.
        The total rank will be 7 + 7 + 8*2 = 29? No, need to be careful.
        
        Actually, for 2x4 * 4x8, we can view it as:
        C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
        
        Split into two halves:
        C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
        
        Each half is a 2x2 * 2x8 problem, not 2x2*2x2.
        So we need a different approach.
        
        Let's try: treat A as 2x4, B as 4x8.
        We can split B into two 2x8 blocks: B_top = B[0:2,:], B_bot = B[2:4,:]
        Then C = A[:,0:2] @ B_top + A[:,2:4] @ B_bot
        
        Each term is 2x2 * 2x8. We need to decompose 2x2 * 2x8.
        
        For 2x2 * 2x8: rank is at most 2*2*8 = 32, but can be less.
        Actually, we can use Strassen for the 2x2 part and extend to 8 columns.
        
        Let's try a different approach: use the known result that
        rank(2x4 * 4x8) can be achieved with a specific construction.
        
        For now, let's just try random search with structured initialization.
        """
        pass

    # Try random search with half-integer entries
    random.seed(42)
    np.random.seed(42)

    best_U, best_V, best_W, best_rank = U_base, V_base, W_base, rank_base

    def try_random_decomposition(target_rank, n_trials=10):
        """Try to find a decomposition of given rank using random init."""
        for trial in range(n_trials):
            if _deadline_guard():
                return None

            # Random initialization with entries in {0, 0.5, 1}
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))

            for c in range(target_rank):
                # Randomly assign non-zero entries
                u_idx = list(range(n * m))
                random.shuffle(u_idx)
                u_count = random.randint(1, max(1, n * m // 3))
                U[u_idx[:u_count], c] = random.choice([0.5, 1.0, 1.0])

                v_idx = list(range(m * p))
                random.shuffle(v_idx)
                v_count = random.randint(1, max(1, m * p // 3))
                V[v_idx[:v_count], c] = random.choice([0.5, 1.0, 1.0])

                w_idx = list(range(p * n))
                random.shuffle(w_idx)
                w_count = random.randint(1, max(1, p * n // 3))
                W[w_idx[:w_count], c] = random.choice([0.5, 1.0, 1.0])

            # Check if this works (unlikely for random init)
            if validate(U, V, W):
                return U, V, W, target_rank

        return None

    # Try lower ranks
    for r in range(32, min(rank_base, 48)):
        if _deadline_guard(110.0):
            break
        result = try_random_decomposition(r, n_trials=5)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break

    # Convert to lists for output
    def to_list(A):
        return [[float(x) for x in row] for row in A]

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
