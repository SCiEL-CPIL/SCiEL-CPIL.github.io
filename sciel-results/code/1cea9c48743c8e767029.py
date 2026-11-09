import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    best_rank = n * m * p
    best_factors = None
    
    # Initialize Schoolbook
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def get_supports(U, V, W):
        # Returns list of tuples (support_U, support_V, support_W) for each column
        supports = []
        for c in range(U.shape[1]):
            su = tuple(np.nonzero(U[:, c])[0])
            sv = tuple(np.nonzero(V[:, c])[0])
            sw = tuple(np.nonzero(W[:, c])[0])
            supports.append((su, sv, sw))
        return supports

    def merge_factors(U, V, W):
        # Greedy merge of columns with identical supports
        changed = True
        while changed:
            changed = False
            supports = get_supports(U, V, W)
            merged = False
            # To make it faster, we can use a dictionary
            support_to_indices = {}
            for i, sup in enumerate(supports):
                if sup not in support_to_indices:
                    support_to_indices[sup] = []
                support_to_indices[sup].append(i)
            
            for sup, indices in support_to_indices.items():
                if len(indices) > 1:
                    # Merge all these columns into the first one
                    first = indices[0]
                    for other in indices[1:]:
                        U[:, first] += U[:, other]
                        V[:, first] += V[:, other]
                        W[:, first] += W[:, other]
                    # Remove the others
                    # We need to carefully remove columns from numpy arrays
                    # It's easier to reconstruct the arrays
                    keep = [i for i in range(U.shape[1]) if i not in indices[1:]]
                    U = U[:, keep]
                    V = V[:, keep]
                    W = W[:, keep]
                    changed = True
                    merged = True
                    break # Restart scanning after a merge
        
        return U, V, W

    def random_transform(U, V, W, num_transforms=10):
        # Apply random linear combinations to random pairs of columns
        # This helps find new identical supports
        for _ in range(num_transforms):
            if U.shape[1] < 2: break
            i, j = random.sample(range(U.shape[1]), 2)
            # u_i = u_i + alpha * u_j
            # To keep integers/half-integers, we can use alpha = 1 or -1 or 0.5
            # But random float alpha might break exactness if not careful.
            # However, the problem allows half-integers.
            # If we use random floats, we might lose exactness.
            # Better: use integer linear combinations.
            alpha = random.choice([-1, 1, 2, -2])
            U[:, i] += alpha * U[:, j]
            V[:, i] += alpha * V[:, j]
            W[:, i] += alpha * W[:, j]
        return U, V, W

    # Main search loop
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        U, V, W = get_schoolbook()
        
        # Randomize
        U, V, W = random_transform(U, V, W, num_transforms=random.randint(10, 50))
        
        # Merge
        U, V, W = merge_factors(U, V, W)
        
        current_rank = U.shape[1]
        if current_rank < best_rank:
            best_rank = current_rank
            best_factors = (U.copy(), V.copy(), W.copy())
            # If we reach a very low rank, we might stop early? 
            # Theoretical lower bound for 3x4x5 is not 60.
            # Let's just keep searching for the best.

    # Convert to list of lists for output
    # Ensure entries are integer or half-integer
    # Multiply by 2 to check if integer, then divide back if needed?
    # The problem says "integer/half-integer entries".
    # Our operations started with integers and added integers, so they are integers.
    # If we used alpha=0.5, they would be half-integers.
    # We used integers in random_transform, so they are integers.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
