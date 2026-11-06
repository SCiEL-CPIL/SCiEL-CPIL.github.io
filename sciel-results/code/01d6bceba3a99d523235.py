# EVOLVE-BLOCK-START
import time
import random

def construct():
    base = 51975
    deadline = time.time() + 115
    start = time.time()

    # divisors > 1 of 51975 = 3^3*5^2*7*11
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()
    n_div = len(divisors)

    # Precompute for each divisor m: list of residue positions
    # For fast marginal gain: count how many uncovered residues a class covers
    # Use coverage count array approach
    
    def score_from_coverage(covered_counts):
        """Count residues with coverage 0."""
        return sum(1 for c in covered_counts if c == 0)
    
    def greedy(order):
        """Greedy construction: pick residue maximizing marginal coverage."""
        # covered_counts[x] = number of classes covering residue x
        covered_counts = [0] * base
        chosen = {}
        for m in order:
            best_r = 0
            best_gain = -1
            for r in range(m):
                gain = 0
                for x in range(r, base, m):
                    if covered_counts[x] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            chosen[m] = best_r
            for x in range(best_r, base, m):
                covered_counts[x] += 1
        uncovered = score_from_coverage(covered_counts)
        return chosen, uncovered, covered_counts
    
    def local_search_fast(chosen, covered_counts, deadline):
        """Fast local search using marginal gain computation."""
        n = n_div
        best_score = score_from_coverage(covered_counts)
        improved = True
        passes = 0
        while improved and passes < 80 and time.time() < deadline:
            improved = False
            passes += 1
            for i in range(n):
                if time.time() > deadline:
                    break
                m = divisors[i]
                cur_r = chosen[m]
                # Compute marginal gain of current residue
                cur_gain = 0
                for x in range(cur_r, base, m):
                    if covered_counts[x] == 1:
                        cur_gain += 1
                # Try all other residues
                best_r = cur_r
                best_gain = cur_gain
                for nr in range(m):
                    if nr == cur_r:
                        continue
                    gain = 0
                    for x in range(nr, base, m):
                        if covered_counts[x] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_r = nr
                if best_r != cur_r:
                    # Apply change
                    for x in range(cur_r, base, m):
                        covered_counts[x] -= 1
                    for x in range(best_r, base, m):
                        covered_counts[x] += 1
                    chosen[m] = best_r
                    new_score = score_from_coverage(covered_counts)
                    if new_score < best_score:
                        best_score = new_score
                        improved = True
                        if best_score == 0:
                            return chosen, best_score, covered_counts
                    else:
                        # Revert
                        for x in range(best_r, base, m):
                            covered_counts[x] -= 1
                        for x in range(cur_r, base, m):
                            covered_counts[x] += 1
                        chosen[m] = cur_r
        return chosen, best_score, covered_counts
    
    def drop_and_readd(chosen, covered_counts, deadline):
        """Try dropping one class and re-adding the best available class."""
        n = n_div
        best_score = score_from_coverage(covered_counts)
        
        for _ in range(100):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                mi = divisors[i]
                ri = chosen[mi]
                # Remove class i
                for x in range(ri, base, mi):
                    covered_counts[x] -= 1
                del chosen[mi]
                
                # Find best class to add (any modulus not currently used)
                best_gain = -1
                best_m = None
                best_r = None
                used_moduli = set(chosen.keys())
                for m in divisors:
                    if m in used_moduli:
                        continue
                    for r in range(m):
                        gain = 0
                        for x in range(r, base, m):
                            if covered_counts[x] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_m = m
                            best_r = r
                
                if best_m is None or best_gain < 0:
                    # Restore
                    chosen[mi] = ri
                    for x in range(ri, base, mi):
                        covered_counts[x] += 1
                    continue
                
                # Add best class
                chosen[best_m] = best_r
                for x in range(best_r, base, best_m):
                    covered_counts[x] += 1
                new_score = score_from_coverage(covered_counts)
                
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                    if best_score == 0:
                        return chosen, best_score, covered_counts
                else:
                    # Revert: remove best_m, restore mi
                    for x in range(best_r, base, best_m):
                        covered_counts[x] -= 1
                    del chosen[best_m]
                    chosen[mi] = ri
                    for x in range(ri, base, mi):
                        covered_counts[x] += 1
            
            if not improved:
                break
        
        return chosen, best_score, covered_counts
    
    # --- Main search ---
    best_classes = None
    best_score = base
    best_covered = None

    # Greedy in increasing order (small moduli first - they cover more)
    chosen, sc, cov = greedy(divisors)
    if sc < best_score:
        best_score = sc
        best_classes = dict(chosen)
        best_covered = cov[:]

    # Greedy in decreasing order
    chosen, sc, cov = greedy(divisors[::-1])
    if sc < best_score:
        best_score = sc
        best_classes = dict(chosen)
        best_covered = cov[:]

    # Random orders
    random.seed(12345)
    t_end_rand = deadline - 30
    tries = 0
    while time.time() < t_end_rand and tries < 200:
        order = divisors[:]
        random.shuffle(order)
        chosen, sc, cov = greedy(order)
        if sc < best_score:
            best_score = sc
            best_classes = dict(chosen)
            best_covered = cov[:]
            if best_score == 0:
                break
        tries += 1

    # Local search polish
    if time.time() < deadline - 5 and best_classes is not None:
        chosen, sc, cov = local_search_fast(best_classes, best_covered, deadline - 5)
        if sc < best_score:
            best_score = sc
            best_classes = chosen
            best_covered = cov

    # Drop and re-add
    if time.time() < deadline - 5 and best_score > 0 and best_classes is not None:
        chosen, sc, cov = drop_and_readd(best_classes, best_covered, deadline - 5)
        if sc < best_score:
            best_score = sc
            best_classes = chosen
            best_covered = cov

    # Another round of local search after drop/readd
    if time.time() < deadline - 2 and best_score > 0 and best_classes is not None:
        chosen, sc, cov = local_search_fast(best_classes, best_covered, deadline - 2)
        if sc < best_score:
            best_score = sc
            best_classes = chosen
            best_covered = cov

    # Second round: more random + local if time
    if time.time() < deadline - 20 and best_score > 0:
        random.seed(99999)
        for _ in range(30):
            if time.time() > deadline - 20:
                break
            order = divisors[:]
            random.shuffle(order)
            chosen, sc, cov = greedy(order)
            if sc <= best_score:
                chosen, sc2, cov2 = local_search_fast(chosen, cov, deadline - 20)
                if sc2 < best_score:
                    best_score = sc2
                    best_classes = chosen
                    best_covered = cov2
                    if best_score == 0:
                        break

    # Final local search
    if time.time() < deadline and best_score > 0 and best_classes is not None:
        chosen, sc, cov = local_search_fast(best_classes, best_covered, deadline)
        if sc < best_score:
            best_score = sc
            best_classes = chosen
            best_covered = cov

    if best_classes is None:
        best_classes = {m: 0 for m in divisors}

    return [[m, best_classes[m]] for m in divisors]
# EVOLVE-BLOCK-END
