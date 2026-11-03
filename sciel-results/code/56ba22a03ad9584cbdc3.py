import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110

    if q != 3 or n != 8 or radius != 1:
        return _generic_greedy(q, n, radius)

    total_words = 3 ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute balls and neighbors
    balls = []
    neighbors = []
    for w in words:
        ball = [w]
        nbrs = []
        for i in range(n):
            for d in range(q):
                if d != w[i]:
                    new_w = list(w)
                    new_w[i] = d
                    tw = tuple(new_w)
                    ball.append(tw)
                    nbrs.append(word_to_idx[tw])
        balls.append([word_to_idx[b] for b in ball])
        neighbors.append(nbrs)
        
    # Convert balls to sets for O(1) lookup
    balls_set = [set(b) for b in balls]

    # Initialize with Tetracode product
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_code.append(tuple(h + list(tail)))
            
    code = [word_to_idx[c] for c in initial_code]
    
    coverage_count = [0] * total_words
    for c in code:
        for w in balls[c]:
            coverage_count[w] += 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Delete redundant
        for i in range(len(code)):
            c = code[i]
            has_unique = False
            for w in balls[c]:
                if coverage_count[w] == 1:
                    has_unique = True
                    break
            if not has_unique:
                for w in balls[c]:
                    coverage_count[w] -= 1
                code.pop(i)
                improved = True
                break
                
        if improved:
            continue

        # 2. Shift
        # Calculate U for all
        U = {}
        for c in code:
            U[c] = [w for w in balls[c] if coverage_count[w] == 1]
            
        best_shift = None
        best_gain = -1
        
        for c in code:
            if not U[c]:
                continue
            for c_prime in neighbors[c]:
                # Check validity
                valid = True
                for w in U[c]:
                    if c_prime not in balls_set[w]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Calculate gain
                gain = 0
                for d in code:
                    if d == c:
                        continue
                    if not U[d]:
                        continue
                    subset_ok = True
                    for w in U[d]:
                        if c_prime not in balls_set[w]:
                            subset_ok = False
                            break
                    if subset_ok:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_shift = (c, c_prime)
                    
        if best_shift and best_gain > 0:
            c, c_prime = best_shift
            for w in balls[c]:
                coverage_count[w] -= 1
            for w in balls[c_prime]:
                coverage_count[w] += 1
            idx = code.index(c)
            code[idx] = c_prime
            improved = True

    # Final cleanup
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(code)):
            c = code[i]
            has_unique = False
            for w in balls[c]:
                if coverage_count[w] == 1:
                    has_unique = True
                    break
            if not has_unique:
                for w in balls[c]:
                    coverage_count[w] -= 1
                code.pop(i)
                changed = True
                break

    return [list(words[c]) for c in code]

def _generic_greedy(q, n, radius):
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    balls = []
    for w in words:
        ball = [w]
        for r in range(1, radius + 1):
            for perm in itertools.combinations(range(n), r):
                for deltas in itertools.product(range(1, q), repeat=r):
                    new_w = list(w)
                    for i, d in zip(perm, deltas):
                        new_w[i] = (new_w[i] + d) % q
                    ball.append(tuple(new_w))
        balls.append([word_to_idx[b] for b in ball])
        
    uncovered = [True] * len(words)
    code = []
    while any(uncovered):
        best_c = -1
        max_cov = -1
        for c in range(len(words)):
            cov = sum(1 for w in balls[c] if uncovered[w])
            if cov > max_cov:
                max_cov = cov
                best_c = c
        code.append(best_c)
        for w in balls[best_c]:
            uncovered[w] = False
    return [list(words[c]) for c in code]
