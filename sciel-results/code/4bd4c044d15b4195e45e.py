import itertools
import time

def construct(q, n, radius):
    # For this task, q=3, n=8, radius=1
    # But we should handle general inputs if possible, or assume the specific task constraints.
    # The prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # But the specific task is K_3(8,1).
    # I will implement the specific logic for q=3, n=8, radius=1 as it is the only one we know the seed for.
    # If q,n differ, fallback to the seed or raise.
    # Given the "Public Task Contract", it implies the evaluator *will* pass the specific task params.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to seed logic for other params if any (unlikely)
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    start_time = time.time()
    
    # 1. Generate initial 729 codewords
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    # 2. Compute coverage counts
    # Map tuple to index: idx = w[0]*3^7 + ... + w[7]
    def to_idx(w):
        idx = 0
        pow3 = 3**7
        for val in w:
            idx += val * pow3
            pow3 //= 3
        return idx

    def from_idx(idx):
        w = [0]*8
        pow3 = 3**7
        for i in range(7, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w

    # Precompute neighbors for all 6561 words?
    # That's 6561 * 25 = 164k entries. Fast enough.
    # Or compute on the fly. On the fly is simpler.
    
    cov = [0] * (3**8)
    
    # Add all balls
    for c in codewords:
        ci = to_idx(c)
        cov[ci] += 1
        # Neighbors
        for i in range(8):
            orig = c[i]
            for val in range(3):
                if val == orig: continue
                c[i] = val
                ni = to_idx(c)
                cov[ni] += 1
            c[i] = orig
            
    # 3. Greedy removal
    # We need to be careful with modifying the list while iterating.
    # Use a list of active indices? Or just a set of active codewords?
    # Since codewords are distinct tuples, we can use a list and pop, or a set.
    # Using a list and iterating with indices is safer.
    
    active = list(range(len(codewords)))
    removed = [False] * len(codewords)
    
    # Helper to get neighbors of a word
    def get_neighbors(w):
        # Returns list of indices in cov array
        idxs = [to_idx(w)]
        orig = w[:]
        for i in range(8):
            o = orig[i]
            for v in range(3):
                if v == o: continue
                w[i] = v
                idxs.append(to_idx(w))
            w[i] = o
        return idxs

    while True:
        changed = False
        # Iterate in reverse to allow removal without index shifting issues?
        # Or just iterate and mark.
        # Marking is better.
        for i in range(len(active)):
            if time.time() - start_time > 100:
                break
            
            ci = active[i]
            if removed[ci]:
                continue
                
            c = codewords[ci]
            neigh = get_neighbors(c)
            
            # Check if we can remove
            # A word w in neigh is covered by count cov[w].
            # If we remove c, cov[w] becomes cov[w]-1.
            # We can remove c if cov[w] > 1 for all w in neigh.
            # i.e. min(cov[w] for w in neigh) > 1.
            
            min_cov = 999999
            for idx in neigh:
                if cov[idx] < min_cov:
                    min_cov = cov[idx]
            
            if min_cov > 1:
                # Remove
                for idx in neigh:
                    cov[idx] -= 1
                removed[ci] = True
                changed = True
            else:
                # Cannot remove
                pass
        
        if not changed:
            break
            
    # Construct result
    result = [codewords[i] for i in range(len(codewords)) if not removed[i]]
    return result
