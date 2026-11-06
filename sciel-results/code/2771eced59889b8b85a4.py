import time
import random

def construct():
    n = 256
    max_size = 64
    beam_width = 200  # Increase beam width, rely on deduplication
    num_restarts = 5  # Multiple random orders
    
    def is_valid_add_A(A, sums_A, diffs_A, diffs_B, x):
        if len(A) >= max_size:
            return False
        if (2 * x) in sums_A:
            return False
        for a in A:
            if (a + x) in sums_A:
                return False
        for a in A:
            d = x - a
            if d != 0 and d in diffs_B:
                return False
        return True

    def is_valid_add_B(B, sums_B, diffs_A, diffs_B, x):
        if len(B) >= max_size:
            return False
        if (2 * x) in sums_B:
            return False
        for b in B:
            if (b + x) in sums_B:
                return False
        for b in B:
            d = x - b
            if d != 0 and d in diffs_A:
                return False
        return True

    def add_to_A(A, sums_A, diffs_A, x):
        new_A = A + [x]
        new_sums_A = set(sums_A)
        new_diffs_A = set(diffs_A)
        for a in A:
            new_sums_A.add(a + x)
            d = x - a
            if d != 0:
                new_diffs_A.add(d)
                new_diffs_A.add(-d)
        new_sums_A.add(2 * x)
        return new_A, new_sums_A, new_diffs_A

    def add_to_B(B, sums_B, diffs_B, x):
        new_B = B + [x]
        new_sums_B = set(sums_B)
        new_diffs_B = set(diffs_B)
        for b in B:
            new_sums_B.add(b + x)
            d = x - b
            if d != 0:
                new_diffs_B.add(d)
                new_diffs_B.add(-d)
        new_sums_B.add(2 * x)
        return new_B, new_sums_B, new_diffs_B

    def run_beam_search(order):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        beam = [(A, B, sums_A, sums_B, diffs_A, diffs_B)]
        seen_states = set()
        
        for x in order:
            new_beam = []
            new_seen = set()
            
            for (A, B, sums_A, sums_B, diffs_A, diffs_B) in beam:
                # Option 1: Skip x
                key = (tuple(A), tuple(B))
                if key not in new_seen:
                    new_seen.add(key)
                    new_beam.append((A, B, sums_A, sums_B, diffs_A, diffs_B))
                
                # Option 2: Add x to A
                if is_valid_add_A(A, sums_A, diffs_A, diffs_B, x):
                    nA, nSumsA, nDiffsA = add_to_A(A, sums_A, diffs_A, x)
                    key = (tuple(nA), tuple(B))
                    if key not in new_seen:
                        new_seen.add(key)
                        new_beam.append((nA, B, nSumsA, sums_B, nDiffsA, diffs_B))
                
                # Option 3: Add x to B
                if is_valid_add_B(B, sums_B, diffs_A, diffs_B, x):
                    nB, nSumsB, nDiffsB = add_to_B(B, sums_B, diffs_B, x)
                    key = (tuple(A), tuple(nB))
                    if key not in new_seen:
                        new_seen.add(key)
                        new_beam.append((A, nB, sums_A, nSumsB, diffs_A, nDiffsB))
                
                # Option 4: Add x to Both
                if is_valid_add_A(A, sums_A, diffs_A, diffs_B, x):
                    nA, nSumsA, nDiffsA = add_to_A(A, sums_A, diffs_A, x)
                    if is_valid_add_B(B, sums_B, nDiffsA, diffs_B, x):
                        nB, nSumsB, nDiffsB = add_to_B(B, sums_B, diffs_B, x)
                        key = (tuple(nA), tuple(nB))
                        if key not in new_seen:
                            new_seen.add(key)
                            new_beam.append((nA, nB, nSumsA, nSumsB, nDiffsA, nDiffsB))
            
            # Prune beam
            if not new_beam:
                break
                
            # Sort by size, then random tie-breaker to keep diversity
            new_beam.sort(key=lambda s: len(s[0]) + len(s[1]), reverse=True)
            
            # Keep top beam_width
            beam = new_beam[:beam_width]
            
        if beam:
            best_state = max(beam, key=lambda s: len(s[0]) + len(s[1]))
            return best_state[0], best_state[1]
        else:
            return [], []

    start_time = time.time()
    best_A, best_B = [], []
    best_score = 0
    
    # Try multiple random orders
    for i in range(num_restarts):
        if time.time() - start_time > 55:
            break
            
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        A, B = run_beam_search(order)
        score = len(A) + len(B)
        
        if score > best_score:
            best_score = score
            best_A, best_B = A, B
            
    if best_score == 0:
        return [[1], [2]]
        
    return [best_A, best_B]
