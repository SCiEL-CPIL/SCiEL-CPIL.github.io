"""Odd covering systems with moduli dividing 51975; improved local search + SA."""
import time
import random
import numpy as np

BASE = 51975

# Precompute all odd divisors > 1 of 51975
_DIVS = []
for _d in range(3, BASE + 1, 2):
    if BASE % _d == 0:
        _DIVS.append(_d)
_DIVS.sort()
N_DIVS = len(_DIVS)

# Precompute position arrays for each divisor and each residue
# pos_cache[m][r] = np.array of positions r, r+m, r+2m, ... < BASE
_pos_cache = {}
for m in _DIVS:
    arr = []
    for r in range(m):
        arr.append(np.arange(r, BASE, m))
    _pos_cache[m] = arr

def _score_from_cov(cov_count):
    return int(np.sum(cov_count == 0))

def _greedy(order, rng=None):
    """Greedy construction: for each modulus in order, pick the residue covering most uncovered."""
    covered = np.zeros(BASE, dtype=np.uint8)
    classes = []
    for m in order:
        best_r = 0
        best_c = -1
        for r in range(m):
            positions = _pos_cache[m][r]
            c = int(np.sum(covered[positions] == 0))
            if c > best_c:
                best_c = c
                best_r = r
            if best_c == len(positions):
                break
        classes.append([m, best_r])
        covered[_pos_cache[m][best_r]] = 1
    return classes

def _build_cov(classes):
    cov = np.zeros(BASE, dtype=np.int16)
    for m, r in classes:
        cov[_pos_cache[m][r]] += 1
    return cov

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s kill
    rng = random.Random(42)

    # Try multiple greedy orderings and pick the best
    best_classes = None
    best_score = BASE  # worst case
    best_cov = None

    # Ordering 1: increasing modulus
    classes1 = _greedy(_DIVS, rng)
    cov1 = _build_cov(classes1)
    s1 = _score_from_cov(cov1)
    if s1 < best_score:
        best_score = s1
        best_classes = [list(c) for c in classes1]
        best_cov = cov1.copy()

    # Ordering 2: decreasing modulus
    classes2 = _greedy(list(reversed(_DIVS)), rng)
    cov2 = _build_cov(classes2)
    s2 = _score_from_cov(cov2)
    if s2 < best_score:
        best_score = s2
        best_classes = [list(c) for c in classes2]
        best_cov = cov2.copy()

    # Ordering 3: random
    for _ in range(3):
        order = list(_DIVS)
        rng.shuffle(order)
        classes3 = _greedy(order, rng)
        cov3 = _build_cov(classes3)
        s3 = _score_from_cov(cov3)
        if s3 < best_score:
            best_score = s3
            best_classes = [list(c) for c in classes3]
            best_cov = cov3.copy()

    cur_classes = [list(c) for c in best_classes]
    cur_score = best_score
    n = len(cur_classes)

    # Local search with simulated annealing
    temp = 10.0
    cool_rate = 0.995
    min_temp = 0.1

    no_improve = 0
    max_no_improve = 50

    while time.time() < deadline:
        if cur_score == 0:
            return cur_classes

        improved = False
        idx_order = list(range(n))
        rng.shuffle(idx_order)

        # Phase 1: Single residue changes
        for i in idx_order:
            if time.time() > deadline:
                break
            m_i, r_i = cur_classes[i]
            old_pos = _pos_cache[m_i][r_i]
            
            best_r = r_i
            best_sc = cur_score
            
            # Sample residues if m_i is large, otherwise try all
            if m_i > 200:
                candidates = [r_i] + rng.sample(range(m_i), min(50, m_i - 1))
            else:
                candidates = [r for r in range(m_i) if r != r_i]
            
            for r in candidates:
                if r == r_i:
                    continue
                new_pos = _pos_cache[m_i][r]
                # Compute delta: positions that change coverage
                # Remove old, add new
                best_cov[old_pos] -= 1
                best_cov[new_pos] += 1
                sc = _score_from_cov(best_cov)
                best_cov[new_pos] -= 1
                best_cov[old_pos] += 1
                
                if sc < best_sc:
                    best_sc = sc
                    best_r = r
                elif sc == best_sc and rng.random() < 0.1:
                    best_r = r  # random tie-breaking to explore
            
            if best_r != r_i:
                new_pos = _pos_cache[m_i][best_r]
                best_cov[old_pos] -= 1
                best_cov[new_pos] += 1
                cur_classes[i][1] = best_r
                cur_score = best_sc
                improved = True
                no_improve = 0
                if cur_score < best_score:
                    best_score = cur_score
                    best_classes = [list(c) for c in cur_classes]

        # Phase 2: Pairwise residue swaps (only if not too expensive)
        if time.time() < deadline and cur_score > 0:
            pairs_tried = 0
            max_pairs = 200 if n > 20 else 500
            for i in idx_order:
                if time.time() > deadline or pairs_tried >= max_pairs:
                    break
                for j in idx_order:
                    if j <= i or pairs_tried >= max_pairs:
                        continue
                    pairs_tried += 1
                    m_i, r_i = cur_classes[i]
                    m_j, r_j = cur_classes[j]
                    
                    # Swap residues (using modulo to keep valid)
                    new_r_i = r_j % m_i
                    new_r_j = r_i % m_j
                    
                    pos_i_old = _pos_cache[m_i][r_i]
                    pos_j_old = _pos_cache[m_j][r_j]
                    pos_i_new = _pos_cache[m_i][new_r_i]
                    pos_j_new = _pos_cache[m_j][new_r_j]
                    
                    best_cov[pos_i_old] -= 1
                    best_cov[pos_j_old] -= 1
                    best_cov[pos_i_new] += 1
                    best_cov[pos_j_new] += 1
                    sc = _score_from_cov(best_cov)
                    
                    best_cov[pos_i_new] -= 1
                    best_cov[pos_j_new] -= 1
                    best_cov[pos_i_old] += 1
                    best_cov[pos_j_old] += 1
                    
                    if sc < cur_score:
                        # Apply
                        cur_classes[i][1] = new_r_i
                        cur_classes[j][1] = new_r_j
                        cur_score = sc
                        improved = True
                        no_improve = 0
                        if cur_score < best_score:
                            best_score = cur_score
                            best_classes = [list(c) for c in cur_classes]
                    elif sc == cur_score and rng.random() < 0.05:
                        # Accept equal with small probability
                        cur_classes[i][1] = new_r_i
                        cur_classes[j][1] = new_r_j
                        improved = True

        # Phase 3: Simulated annealing - accept worse moves
        if time.time() < deadline and cur_score > 0 and temp > min_temp:
            for i in idx_order:
                if time.time() > deadline:
                    break
                m_i, r_i = cur_classes[i]
                # Pick a random new residue
                new_r = rng.randint(0, m_i - 1)
                if new_r == r_i:
                    continue
                
                old_pos = _pos_cache[m_i][r_i]
                new_pos = _pos_cache[m_i][new_r]
                
                best_cov[old_pos] -= 1
                best_cov[new_pos] += 1
                sc = _score_from_cov(best_cov)
                
                delta = sc - cur_score
                if delta < 0 or (delta > 0 and rng.random() < np.exp(-delta / temp)):
                    # Accept
                    cur_classes[i][1] = new_r
                    cur_score = sc
                    improved = True
                    no_improve = 0
                    if cur_score < best_score:
                        best_score = cur_score
                        best_classes = [list(c) for c in cur_classes]
                else:
                    # Reject
                    best_cov[new_pos] -= 1
                    best_cov[old_pos] += 1
                
                temp *= cool_rate

        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
            # Restart from best if stuck
            if no_improve % 10 == 0:
                cur_classes = [list(c) for c in best_classes]
                cur_score = best_score
                best_cov = _build_cov(cur_classes)
                temp = max(temp, 5.0)

    return best_classes
