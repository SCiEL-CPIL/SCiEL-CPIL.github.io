import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s for return
    
    N = 128
    
    # Initialize with the seed solution which is known to be valid
    A = np.zeros(N+1, dtype=bool)
    A[64:129] = True
    
    # Get initial counts
    def get_target_counts(A):
        conv = np.convolve(A.astype(int), A.astype(int), mode='full')
        return conv[128:257]
        
    target_counts = get_target_counts(A)
    min_c = np.min(target_counts)
    max_c = np.max(target_counts)
    best_max_c = max_c
    best_A = A.copy()
    
    # List of indices in A for fast sampling
    A_indices = list(np.where(A)[0])
    
    while time.time() < deadline:
        # 1. Ensure coverage
        if min_c < 1:
            uncovered = np.where(target_counts == 0)[0] + 128
            if len(uncovered) == 0:
                break
            s = random.choice(uncovered)
            
            # Find a candidate a such that s-a in A
            # s-a must be in A_indices
            # a = s - b for b in A_indices
            # a must not be in A
            candidates = []
            for b in A_indices:
                a = s - b
                if 0 <= a <= N and not A[a]:
                    candidates.append(a)
            
            if candidates:
                a = random.choice(candidates)
                A[a] = True
                A_indices.append(a)
                target_counts = get_target_counts(A)
                min_c = np.min(target_counts)
                max_c = np.max(target_counts)
            else:
                # Cannot fix with single add. Add random.
                a = random.randint(0, N)
                if not A[a]:
                    A[a] = True
                    A_indices.append(a)
                    target_counts = get_target_counts(A)
                    min_c = np.min(target_counts)
                    max_c = np.max(target_counts)
        
        else:
            # 2. Try to improve max_c
            # Try removal
            if len(A_indices) > 20:
                idx_to_remove = random.choice(A_indices)
                A[idx_to_remove] = False
                A_indices.remove(idx_to_remove)
                
                new_counts = get_target_counts(A)
                new_min_c = np.min(new_counts)
                new_max_c = np.max(new_counts)
                
                if new_min_c < 1:
                    # Revert
                    A[idx_to_remove] = True
                    A_indices.append(idx_to_remove)
                else:
                    if new_max_c < best_max_c:
                        best_max_c = new_max_c
                        best_A = A.copy()
                    min_c, max_c = new_min_c, new_max_c
                    target_counts = new_counts
            
            # Try swap
            if len(A_indices) > 20:
                a = random.choice(A_indices)
                b = random.randint(0, N)
                if not A[b]:
                    A[a] = False
                    A_indices.remove(a)
                    A[b] = True
                    A_indices.append(b)
                    
                    new_counts = get_target_counts(A)
                    new_min_c = np.min(new_counts)
                    new_max_c = np.max(new_counts)
                    
                    if new_min_c < 1 or new_max_c > best_max_c:
                        # Revert
                        A[a] = True
                        A_indices.append(a)
                        A[b] = False
                        A_indices.remove(b)
                    else:
                        if new_max_c < best_max_c:
                            best_max_c = new_max_c
                            best_A = A.copy()
                        min_c, max_c = new_min_c, new_max_c
                        target_counts = new_counts

    return list(np.where(best_A)[0])
