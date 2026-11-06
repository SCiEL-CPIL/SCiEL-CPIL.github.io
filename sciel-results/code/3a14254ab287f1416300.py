import time
import random
import numpy as np

BASE = 51975

# All divisors of 51975 = 3^3 * 5^2 * 7 * 11, greater than 1
_DIVS = []
for a in range(4):          # 3^a
    for b in range(3):      # 5^b
        for c in range(2):  # 7^c
            for d in range(2):  # 11^d
                v = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                if v > 1:
                    _DIVS.append(v)
_DIVS.sort()
N = len(_DIVS)

# Map from modulus value to its index in _DIVS
_DIV_IDX = {m: i for i, m in enumerate(_DIVS)}

# Precompute bitsets for each (divisor, residue)
# Each bitset is a list of 64-bit integers representing coverage of BASE positions
_NUM_WORDS = (BASE + 63) // 64  # 813 words

# _BITSETS[i][r] = list of 64-bit ints for positions r, r+m, r+2m, ... mod BASE
_BITSETS = []
for m in _DIVS:
    lst = []
    for r in range(m):
        bits = [0] * _NUM_WORDS
        pos = r
        while pos < BASE:
            bits[pos >> 6] |= (1 << (pos & 63))
            pos += m
        lst.append(bits)
    _BITSETS.append(lst)


def _bitset_or_into(target, bits):
    """OR bits into target in place."""
    for i in range(_NUM_WORDS):
        target[i] |= bits[i]


def _count_uncovered(covered_bits):
    """Count positions not covered (bits == 0)."""
    count = 0
    for word in covered_bits:
        count += 64 - bin(word).count('1')
    # Subtract padding bits beyond BASE
    padding = _NUM_WORDS * 64 - BASE
    if padding > 0:
        # The last word may have padding bits set to 0, which we'd count as uncovered
        # But we only care about positions < BASE, so we need to mask
        pass
    return count - padding  # padding bits are always 0, so they're counted as uncovered


def _count_uncovered_masked(covered_bits):
    """Count positions not covered, properly masking the last word."""
    count = 0
    for i in range(_NUM_WORDS):
        word = covered_bits[i]
        if i == _NUM_WORDS - 1:
            # Mask out padding bits
            valid_bits = BASE - i * 64
            mask = (1 << valid_bits) - 1
            word &= mask
        count += 64 - bin(word).count('1') if i < _NUM_WORDS - 1 else (valid_bits - bin(word).count('1'))
    return count


def _score_from_bits(covered_bits):
    """Score = number of uncovered positions."""
    count = 0
    for i in range(_NUM_WORDS):
        word = covered_bits[i]
        if i == _NUM_WORDS - 1:
            valid_bits = BASE - i * 64
            mask = (1 << valid_bits) - 1
            word &= mask
            count += valid_bits - bin(word).count('1')
        else:
            count += 64 - bin(word).count('1')
    return count


def _build_covered(classes):
    """Build coverage bitset from a list of classes."""
    covered = [0] * _NUM_WORDS
    for m, r in classes:
        idx = _DIV_IDX[m]
        bits = _BITSETS[idx][r]
        _bitset_or_into(covered, bits)
    return covered


def _full_greedy_dynamic(deadline):
    """Dynamic greedy: repeatedly pick the (m,r) that covers the most new positions."""
    covered = [0] * _NUM_WORDS
    classes = []
    used_moduli = set()
    
    while time.time() < deadline:
        best_gain = 0
        best_m = None
        best_r = None
        
        for i, m in enumerate(_DIVS):
            if m in used_moduli:
                continue
            for r in range(m):
                bits = _BITSETS[i][r]
                # Count how many new positions this would cover
                gain = 0
                for w in range(_NUM_WORDS):
                    new_bits = bits[w] & ~covered[w]
                    if w == _NUM_WORDS - 1:
                        valid_bits = BASE - w * 64
                        mask = (1 << valid_bits) - 1
                        new_bits &= mask
                        gain += bin(new_bits).count('1')
                    else:
                        gain += bin(new_bits).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
        
        if best_m is None or best_gain == 0:
            break
        
        used_moduli.add(best_m)
        classes.append([best_m, best_r])
        idx = _DIV_IDX[best_m]
        _bitset_or_into(covered, _BITSETS[idx][best_r])
    
    return classes, covered


def _local_search_bitset(classes, covered, deadline, rng):
    """Local search using bitset operations."""
    best_score = _score_from_bits(covered)
    best_classes = [c[:] for c in classes]
    best_covered = covered[:]
    n_classes = len(classes)
    
    no_improve_rounds = 0
    max_no_improve = 8
    
    while time.time() < deadline and no_improve_rounds < max_no_improve:
        improved_this_round = False
        
        order = list(range(n_classes))
        rng.shuffle(order)
        
        for ci in order:
            if time.time() >= deadline:
                return best_classes, best_score
            if time.time() > deadline - 1.0:
                break
            
            m, r = classes[ci]
            di = _DIV_IDX[m]
            
            # Remove current coverage
            old_bits = _BITSETS[di][r]
            temp_covered = covered[:]
            for w in range(_NUM_WORDS):
                temp_covered[w] ^= old_bits[w]  # This is wrong; we need to subtract, not XOR
            
            # Actually, let's use a different approach: rebuild without this class
            # For efficiency, let's just compute the score change directly
            
            best_r_new = r
            best_score_candidate = best_score
            
            # Try removing and re-adding
            for new_r in range(m):
                if new_r == r:
                    continue
                
                # Compute coverage without class ci, then with new_r
                # This is expensive; let's approximate by just checking the delta
                new_bits = _BITSETS[di][new_r]
                
                # Count positions covered by old but not new (lost)
                # And positions covered by new but not old (gained)
                # Net change in uncovered = lost - gained (approximately)
                
                # Actually, let's just do the full rebuild for correctness
                if time.time() > deadline - 0.5:
                    break
                
                # Build coverage without this class
                cov_no_ci = [0] * _NUM_WORDS
                for j, (mj, rj) in enumerate(classes):
                    if j == ci:
                        continue
                    idx_j = _DIV_IDX[mj]
                    _bitset_or_into(cov_no_ci, _BITSETS[idx_j][rj])
                
                # Add new_r
                _bitset_or_into(cov_no_ci, _BITSETS[di][new_r])
                
                candidate_score = _score_from_bits(cov_no_ci)
                
                if candidate_score < best_score_candidate:
                    best_score_candidate = candidate_score
                    best_r_new = new_r
            
            if best_r_new != r:
                classes[ci] = [m, best_r_new]
                covered = _build_covered(classes)
                best_score = best_score_candidate
                best_classes = [c[:] for c in classes]
                best_covered = covered[:]
                improved_this_round = True
                
                if best_score == 0:
                    return best_classes, best_score
        
        if improved_this_round:
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
    
    return best_classes, best_score


def _greedy_fixed_order(order, deadline):
    """Greedy with fixed order of moduli."""
    covered = [0] * _NUM_WORDS
    classes = []
    for i in order:
        if time.time() >= deadline:
            break
        m = _DIVS[i]
        best_r = 0
        best_gain = -1
        for r in range(m):
            bits = _BITSETS[i][r]
            gain = 0
            for w in range(_NUM_WORDS):
                new_bits = bits[w] & ~covered[w]
                if w == _NUM_WORDS - 1:
                    valid_bits = BASE - w * 64
                    mask = (1 << valid_bits) - 1
                    new_bits &= mask
                    gain += bin(new_bits).count('1')
                else:
                    gain += bin(new_bits).count('1')
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes.append([m, best_r])
        _bitset_or_into(covered, _BITSETS[i][best_r])
    return classes, covered


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    best_classes = None
    best_score = BASE + 1
    rng = random.Random(42)
    
    # Phase 1: Dynamic greedy
    classes, covered = _full_greedy_dynamic(deadline - 60)
    s = _score_from_bits(covered)
    if s < best_score:
        best_score = s
        best_classes = [c[:] for c in classes]
    
    # Phase 2: Fixed-order greedy with different orderings
    order_asc = list(range(N))
    classes, covered = _greedy_fixed_order(order_asc, deadline - 50)
    s = _score_from_bits(covered)
    if s < best_score:
        best_score = s
        best_classes = [c[:] for c in classes]
    
    order_desc = list(range(N - 1, -1, -1))
    classes, covered = _greedy_fixed_order(order_desc, deadline - 40)
    s = _score_from_bits(covered)
    if s < best_score:
        best_score = s
        best_classes = [c[:] for c in classes]
    
    # Phase 3: Randomized greedy + local search
    while time.time() < deadline - 30:
        order = list(range(N))
        rng.shuffle(order)
        classes, covered = _greedy_fixed_order(order, deadline - 25)
        s = _score_from_bits(covered)
        
        if s < best_score:
            best_score = s
            best_classes = [c[:] for c in classes]
            if best_score == 0:
                return best_classes
        
        # Local search if promising
        if s <= best_score + 100 and time.time() < deadline - 10:
            ls_deadline = min(deadline - 5.0, time.time() + 5.0)
            improved_classes, improved_score = _local_search_bitset(
                [c[:] for c in classes], covered[:], ls_deadline, rng
            )
            if improved_score < best_score:
                best_score = improved_score
                best_classes = improved_classes
                if best_score == 0:
                    return best_classes
    
    # Phase 4: Final local search on best
    if best_classes is not None and time.time() < deadline - 3.0:
        covered = _build_covered(best_classes)
        ls_deadline = min(deadline - 1.0, time.time() + 10.0)
        if ls_deadline > time.time() + 1.0:
            best_classes, best_score = _local_search_bitset(
                best_classes, covered, ls_deadline, rng
            )
    
    return best_classes
