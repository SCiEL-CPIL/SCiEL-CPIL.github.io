import itertools
import random
import time

def construct(q, n, radius):
    # Specific solver for the ternary covering code K_3(7,1)
    if q == 3 and n == 7 and radius == 1:
        total_words = 3 ** n
        all_words = list(itertools.product(range(3), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        covers = [[] for _ in range(total_words)]
        covered_by = [[] for _ in range(total_words)]
        
        for i, w in enumerate(all_words):
            covers[i].append(i)
            covered_by[i].append(i)
            for pos in range(n):
                for val in range(3):
                    if val == w[pos]:
                        continue
                    neighbor = list(w)
                    neighbor[pos] = val
                    neighbor = tuple(neighbor)
                    j = word_to_idx[neighbor]
                    covers[i].append(j)
                    covered_by[j].append(i)
        
        def greedy_solve(seed):
            rng = random.Random(seed)
            covered = [False] * total_words
            remaining = [len(covers[i]) for i in range(total_words)]
            code = []
            uncovered_count = total_words
            
            while uncovered_count > 0:
                max_rem = -1
                best_candidates = []
                for i in range(total_words):
                    r = remaining[i]
                    if r > max_rem:
                        max_rem = r
                        best_candidates = [i]
                    elif r == max_rem:
                        best_candidates.append(i)
                
                if max_rem <= 0:
                    break
                    
                chosen = best_candidates[rng.randint(0, len(best_candidates) - 1)]
                code.append(chosen)
                
                for u in covers[chosen]:
                    if not covered[u]:
                        covered[u] = True
                        uncovered_count -= 1
                    for v in covered_by[u]:
                        remaining[v] -= 1
                        
            return code

        best_code = None
        best_size = float('inf')
        
        start_time = time.time()
        seed = 0
        # Run until we have used most of the time budget
        while time.time() - start_time < 145:
            code = greedy_solve(seed)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                if best_size <= 186:
                    break
            seed += 1
            
        if best_code is None:
            # Fallback to direct sum
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            
        return [list(all_words[i]) for i in best_code]

    # Fallback for other parameters
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [[0] * n]
