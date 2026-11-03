<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    random.seed(42)
    
    candidates = []
    
    # Exhaustive search for small sets (n=4 to 8)
    if time.time() < deadline - 60:
        ex_U, ex_score = _exhaustive_small_set_search(max_n=8, max_val=25, deadline=deadline - 60)
        if ex_score > best_score:
            best_score = ex_score
            best_U = ex_U
        candidates.append(ex_U)
    
    # Banded DP constructions (new approach)
    for max_p, tsize in [(100, 10), (120, 12), (150, 14), (180, 16)]:
        if time.time() > deadline - 55:
            break
        dp_U, dp_score = _banded_dp_construct_v3(max_pos=max_p, target_size=tsize, deadline=deadline - 55)
        if dp_score > best_score:
            best_score = dp_score
            best_U = dp_U
        candidates.append(dp_U)
    
    # More banded DP with different parameters
    for max_p, tsize in [(80, 8), (100, 10), (130, 13)]:
        if time.time() > deadline - 50:
            break
        dp_U, dp_score = _banded_dp_construct_v2(max_pos=max_p, target_size=tsize, deadline=deadline - 50)
        if dp_score > best_score:
            best_score = dp_score
            best_U = dp_U
        candidates.append(dp_U)
    
    # Golomb rulers of various sizes
    for n in range(3, 35):
        if time.time() > deadline - 50:
            break
        gr = _golomb_ruler(n)
        candidates.append(gr)
    
    # Prime-step constructions with greedy selection
    for num_steps in [6, 10, 14, 18, 22, 26, 30]:
        if time.time() > deadline - 50:
            break
        for max_p in [150, 300, 500]:
            if time.time() > deadline - 50:
                break
            ps = _prime_step_construction(num_steps, max_prime=max_p)
            candidates.append(ps)
    
    # Greedy prime selection
    for _ in range(10):
        if time.time() > deadline - 50:
            break
        for mp in [250, 450]:
            if time.time() > deadline - 50:
                break
            gp = _greedy_prime_selection([0], max_prime=mp, max_steps=40)
            candidates.append(gp)
    
    # Hierarchical staircase constructions
    for nb in range(3, 9):
        if time.time() > deadline - 50:
            break
        hs = _hierarchical_staircase(nb, deadline=deadline - 50)
        candidates.append(hs)
    
    # Random sets with various sizes
    for trial in range(25):
        if time.time() > deadline - 50:
            break
        n = random.randint(4, 40)
        max_val = random.randint(40, 1500)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Structured sets: arithmetic progressions with gaps
    for base in [2, 3, 5, 7, 11, 13, 17]:
        if time.time() > deadline - 50:
            break
        U = [0]
        for i in range(1, 20):
            U.append(i * base)
        candidates.append(U[:min(len(U), 20)])
    
    # Sparse random sets
    for trial in range(15):
        if time.time() > deadline - 50:
            break
        n = random.randint(6, 35)
        max_val = random.randint(150, 3000)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Modular residue constructions
    for max_p in [100, 200, 350]:
        if time.time() > deadline - 45:
            break
        mr_U, mr_score = _modular_residue_construction(max_prime=max_p, max_target_size=50, deadline=deadline - 45)
        if len(mr_U) >= 2:
            candidates.append(mr_U)
    
    # Evaluate candidates and keep top ones
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline - 40:
            break
        s = _score_set(U)
        scored_candidates.append((s, U))
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Local search on best candidate
    if time.time() < deadline - 25:
        best_U, best_score = _local_search(best_U, max_iters=10000, deadline=deadline - 25)
    
    # Try local search on top few other candidates
    for s, U in scored_candidates[:10]:
        if time.time() > deadline - 25:
            break
        if U == best_U:
            continue
        if s > best_score - 0.04:
            improved_U, improved_score = _local_search(U, max_iters=4000, deadline=deadline - 25)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    # Final high-precision polish pass
    if time.time() < deadline - 8:
        best_U, best_score = _polish(best_U, deadline=deadline - 3)
    
    # Verify 0 is in the set and all elements are nonnegative integers
    best_U = [int(x) for x in best_U if int(x) >= 0]
    if 0 not in best_U:
        best_U.append(0)
    best_U = sorted(set(best_U))
    
    return best_U
=======
def _repulsive_chain_construct(max_pos=200, target_size=25, deadline=None):
    """
    Ising-chain-inspired construction: greedily place elements on a linear grid
    to maximize a 'repulsive energy' that rewards unique pairwise differences
    and penalizes sum collisions. Uses incremental scoring for efficiency.
    """
    if deadline is None:
        deadline = time.time() + 20
    
    scorer = _IncrementalScorer([0])
    U = [0]
    best_score = scorer.score()
    
    for step in range(1, target_size):
        if deadline and time.time() > deadline - 2:
            break
        
        # Generate candidate positions
        candidates = set()
        max_existing = max(U) if U else 0
        
        # Positions spread across the range
        for x in range(1, max_pos + 1):
            candidates.add(x)
        
        # Positions near existing elements (for fine-tuning)
        for x in U:
            for offset in range(-10, 11):
                nv = x + offset
                if nv > 0:
                    candidates.add(nv)
        
        # Positions that are sums/diffs of existing elements
        for a in U:
            for b in U:
                candidates.add(a + b)
                if a != b:
                    candidates.add(abs(a - b))
        
        filtered = [c for c in candidates if c > 0 and c not in scorer.U and c <= max_pos]
        
        if not filtered:
            break
        
        # Use incremental try_add_score for fast ranking
        scored = []
        for c in filtered:
            sc = scorer.try_add_score(c)
            scored.append((sc, c))
        
        scored.sort(key=lambda x: -x[0])
        
        # Take the best candidate
        best_sc, best_c = scored[0]
        
        if best_sc > best_score + 1e-12:
            scorer.add(best_c)
            U.append(best_c)
            best_score = best_sc
        else:
            # No improvement possible with this set, break
            break
    
    return sorted(U), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    random.seed(42)
    
    candidates = []
    
    # Exhaustive search for small sets (n=4 to 8)
    if time.time() < deadline - 60:
        ex_U, ex_score = _exhaustive_small_set_search(max_n=8, max_val=25, deadline=deadline - 60)
        if ex_score > best_score:
            best_score = ex_score
            best_U = ex_U
        candidates.append(ex_U)
    
    # Banded DP constructions with incremental scoring
    for max_p, tsize in [(100, 10), (120, 12), (150, 14), (180, 16)]:
        if time.time() > deadline - 55:
            break
        dp_U, dp_score = _banded_dp_construct_v3(max_pos=max_p, target_size=tsize, deadline=deadline - 55)
        if dp_score > best_score:
            best_score = dp_score
            best_U = dp_U
        candidates.append(dp_U)
    
    # More banded DP with different parameters
    for max_p, tsize in [(80, 8), (100, 10), (130, 13)]:
        if time.time() > deadline - 50:
            break
        dp_U, dp_score = _banded_dp_construct_v2(max_pos=max_p, target_size=tsize, deadline=deadline - 50)
        if dp_score > best_score:
            best_score = dp_score
            best_U = dp_U
        candidates.append(dp_U)
    
    # Repulsive chain construction (new Ising-inspired approach)
    for max_p, tsize in [(150, 15), (200, 20), (250, 25), (300, 30)]:
        if time.time() > deadline - 45:
            break
        rc_U, rc_score = _repulsive_chain_construct(max_pos=max_p, target_size=tsize, deadline=deadline - 45)
        if rc_score > best_score:
            best_score = rc_score
            best_U = rc_U
        candidates.append(rc_U)
    
    # Golomb rulers of various sizes
    for n in range(3, 35):
        if time.time() > deadline - 50:
            break
        gr = _golomb_ruler(n)
        candidates.append(gr)
    
    # Prime-step constructions with greedy selection
    for num_steps in [6, 10, 14, 18, 22, 26, 30]:
        if time.time() > deadline - 50:
            break
        for max_p in [150, 300, 500]:
            if time.time() > deadline - 50:
                break
            ps = _prime_step_construction(num_steps, max_prime=max_p)
            candidates.append(ps)
    
    # Greedy prime selection
    for _ in range(10):
        if time.time() > deadline - 50:
            break
        for mp in [250, 450]:
            if time.time() > deadline - 50:
                break
            gp = _greedy_prime_selection([0], max_prime=mp, max_steps=40)
            candidates.append(gp)
    
    # Hierarchical staircase constructions
    for nb in range(3, 9):
        if time.time() > deadline - 50:
            break
        hs = _hierarchical_staircase(nb, deadline=deadline - 50)
        candidates.append(hs)
    
    # Random sets with various sizes
    for trial in range(25):
        if time.time() > deadline - 50:
            break
        n = random.randint(4, 40)
        max_val = random.randint(40, 1500)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Structured sets: arithmetic progressions with gaps
    for base in [2, 3, 5, 7, 11, 13, 17]:
        if time.time() > deadline - 50:
            break
        U = [0]
        for i in range(1, 20):
            U.append(i * base)
        candidates.append(U[:min(len(U), 20)])
    
    # Sparse random sets
    for trial in range(15):
        if time.time() > deadline - 50:
            break
        n = random.randint(6, 35)
        max_val = random.randint(150, 3000)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Modular residue constructions
    for max_p in [100, 200, 350]:
        if time.time() > deadline - 45:
            break
        mr_U, mr_score = _modular_residue_construction(max_prime=max_p, max_target_size=50, deadline=deadline - 45)
        if len(mr_U) >= 2:
            candidates.append(mr_U)
    
    # Evaluate candidates and keep top ones
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline - 40:
            break
        s = _score_set(U)
        scored_candidates.append((s, U))
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Local search on best candidate (now faster with incremental scoring)
    if time.time() < deadline - 25:
        best_U, best_score = _local_search(best_U, max_iters=15000, deadline=deadline - 25)
    
    # Try local search on top few other candidates
    for s, U in scored_candidates[:10]:
        if time.time() > deadline - 25:
            break
        if U == best_U:
            continue
        if s > best_score - 0.04:
            improved_U, improved_score = _local_search(U, max_iters=5000, deadline=deadline - 25)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    # Final high-precision polish pass
    if time.time() < deadline - 8:
        best_U, best_score = _polish(best_U, deadline=deadline - 3)
    
    # Verify 0 is in the set and all elements are nonnegative integers
    best_U = [int(x) for x in best_U if int(x) >= 0]
    if 0 not in best_U:
        best_U.append(0)
    best_U = sorted(set(best_U))
    
    return best_U
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
