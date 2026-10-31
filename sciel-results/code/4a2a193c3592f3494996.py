import numpy as np

N = 50

def construct():
    # Find a Sidon set in range(N)
    def find_sidon_set(n, target_size):
        best = []
        for _ in range(100): # Random restarts
            current = []
            sums = set()
            # Random order of indices
            indices = list(range(n))
            np.random.shuffle(indices)
            for i in indices:
                # Check if i can be added
                can_add = True
                # Check pairs (i, i) and (i, j) for j in current
                # New sums: i+i, i+j
                new_sums = [2*i]
                for j in current:
                    new_sums.append(i+j)
                
                # Check for collisions
                for s in new_sums:
                    if s in sums:
                        can_add = False
                        break
                    # Also check if this new sum collides with another new sum?
                    # If i+j == i+k, then j==k, which is not possible if j,k distinct.
                    # If i+j == 2i, then j=i, not possible.
                    # So we just need to check against existing sums.
                
                if can_add:
                    # Check internal collisions in new_sums?
                    # e.g. i+j1 == i+j2 => j1==j2 (impossible)
                    # i+j == 2i => j=i (impossible)
                    # So no internal collisions.
                    
                    current.append(i)
                    for s in new_sums:
                        sums.add(s)
                    
                    if len(current) >= target_size:
                        break
            
            if len(current) > len(best):
                best = current
            if len(best) >= target_size:
                break
        return best

    # Try to find a large Sidon set
    # Max possible is around sqrt(50) ~ 7-8
    best_sidon = []
    for k in range(10, 0, -1):
        s = find_sidon_set(N, k)
        if len(s) == k:
            best_sidon = s
            break
    
    if not best_sidon:
        # Fallback to a known small set
        best_sidon = [0, 1, 4, 10]

    h = np.zeros(N)
    k = len(best_sidon)
    for idx in best_sidon:
        h[idx] = 1.0 / k
        
    return list(h)
