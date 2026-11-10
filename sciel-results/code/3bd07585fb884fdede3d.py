"""Rewritten program for the suite task; maximizes comparable pairs in a 40-subset family on 12 vertices."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

N = 12
M = 40
TIME_LIMIT = 50.0  # Safety margin before 60s kill

# Global precomputation (done once at module level for speed)
_ALL_MASKS = list(range(1 << N))
_POPCOUNT = [0] * (1 << N)
for _m in range(1, 1 << N):
    _POPCOUNT[_m] = _POPCOUNT[_m >> 1] + (_m & 1)

# Precompute subsets and supersets for all masks
_SUBSETS_OF = [[] for _ in range(1 << N)]
_SUPERSETS_OF = [[] for _ in range(1 << N)]
for _m in range(1 << N):
    # Proper subsets
    if _m > 0:
        _sub = (_m - 1) & _m
        while _sub:
            _SUBSETS_OF[_m].append(_sub)
            _sub = (_sub - 1) & _m
        _SUBSETS_OF[_m].append(0)
    # Proper supersets
    _complement = ((1 << N) - 1) ^ _m
    if _complement:
        _t = _complement
        while _t:
            _s = _m | _t
            _SUPERSETS_OF[_m].append(_s)
            _t = (_t - 1) & _complement

def _is_comparable(a, b):
    """Check if a and b are comparable (one is strict subset of the other)."""
    if a == b:
        return False
    if (a & b) == a:
        return True
    if (a & b) == b:
        return True
    return False

def _count_comparable(family):
    """Count unordered pairs where one is a strict subset of the other."""
    n = len(family)
    count = 0
    for i in range(n):
        mi = family[i]
        for j in range(i + 1, n):
            mj = family[j]
            if _is_comparable(mi, mj):
                count += 1
    return count

def _greedy_construct():
    """Greedy construction: start with empty set, then add sets that maximize new comparable pairs."""
    family = [0]
    in_family = [False] * (1 << N)
    in_family[0] = True
    
    # comparable_count[m] = number of sets in family that are comparable with m
    comparable_count = [0] * (1 << N)
    for m in range(1, 1 << N):
        comparable_count[m] = 1  # comparable with empty set
    
    for _ in range(M - 1):
        best_mask = -1
        best_count = -1
        for m in range(1 << N):
            if not in_family[m] and comparable_count[m] > best_count:
                best_count = comparable_count[m]
                best_mask = m
        
        if best_mask == -1:
            break
            
        family.append(best_mask)
        in_family[best_mask] = True
        
        # Update comparable_count for all other masks
        for s in _SUBSETS_OF[best_mask]:
            if not in_family[s]:
                comparable_count[s] += 1
        for s in _SUPERSETS_OF[best_mask]:
            if not in_family[s]:
                comparable_count[s] += 1
    
    return family

def _local_search(family, time_limit):
    """Local search: try to improve by swapping sets. Returns (best_family, best_score)."""
    start_time = time.time()
    
    family = list(family)
    family_set = set(family)
    
    # Precompute comparable_count for each mask relative to current family
    comparable_count = [0] * (1 << N)
    for m in range(1 << N):
        for f in family:
            if _is_comparable(m, f):
                comparable_count[m] += 1
    
    current_score = _count_comparable(family)
    best_family = list(family)
    best_score = current_score
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try swapping each set in family with a set not in family
        for i, fi in enumerate(family):
            # Compute loss from removing fi: number of family members (excluding fi) comparable with fi
            loss = 0
            for j, fj in enumerate(family):
                if i != j and _is_comparable(fi, fj):
                    loss += 1
            
            # Try adding each mask not in family
            best_gain = -1
            best_new_mask = -1
            for m in range(1 << N):
                if m in family_set:
                    continue
                # Number of sets in (family - {fi}) comparable with m
                # = comparable_count[m] - (1 if fi comparable with m else 0)
                is_comp_with_fi = 1 if _is_comparable(m, fi) else 0
                gain = comparable_count[m] - is_comp_with_fi - loss
                if gain > best_gain:
                    best_gain = gain
                    best_new_mask = m
            
            if best_gain > 0 and best_new_mask != -1:
                # Perform the swap
                family[i] = best_new_mask
                family_set.discard(fi)
                family_set.add(best_new_mask)
                
                # Update comparable_count
                # Remove contributions from fi
                for s in _SUBSETS_OF[fi]:
                    if s in family_set:
                        comparable_count[s] -= 1
                for s in _SUPERSETS_OF[fi]:
                    if s in family_set:
                        comparable_count[s] -= 1
                # Add contributions from best_new_mask
                for s in _SUBSETS_OF[best_new_mask]:
                    if s in family_set:
                        comparable_count[s] += 1
                for s in _SUPERSETS_OF[best_new_mask]:
                    if s in family_set:
                        comparable_count[s] += 1
                
                current_score += best_gain
                if current_score > best_score:
                    best_score = current_score
                    best_family = list(family)
                improved = True
                no_improve_count = 0
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 5:
                break
    
    return best_family, best_score

def _polish(family, time_limit):
    """High-precision polish pass: try 2-swap moves (swap two sets for two others)."""
    start_time = time.time()
    
    family = list(family)
    family_set = set(family)
    
    # Precompute comparable_count
    comparable_count = [0] * (1 << N)
    for m in range(1 << N):
        for f in family:
            if _is_comparable(m, f):
                comparable_count[m] += 1
    
    current_score = _count_comparable(family)
    best_family = list(family)
    best_score = current_score
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try 2-swap: remove fi, fj and add m1, m2
        # This is expensive, so only try promising pairs
        for i in range(M):
            if time.time() - start_time > time_limit:
                break
            fi = family[i]
            # Loss from removing fi
            loss_i = 0
            for j, fj in enumerate(family):
                if i != j and _is_comparable(fi, fj):
                    loss_i += 1
            
            for j in range(i + 1, M):
                if time.time() - start_time > time_limit:
                    break
                fj = family[j]
                # Loss from removing fj (accounting for fi already removed)
                loss_j = 0
                for k, fk in enumerate(family):
                    if k != i and k != j and _is_comparable(fj, fk):
                        loss_j += 1
                # If fi and fj are comparable, we double-counted that pair
                if _is_comparable(fi, fj):
                    total_loss = loss_i + loss_j - 1
                else:
                    total_loss = loss_i + loss_j
                
                # Try adding two new masks
                best_gain = -1
                best_m1 = -1
                best_m2 = -1
                # Sample a subset of candidates for speed
                candidates = [m for m in range(1 << N) if m not in family_set]
                random.shuffle(candidates)
                candidates = candidates[:200]  # Limit candidates
                
                for m1 in candidates:
                    if time.time() - start_time > time_limit:
                        break
                    # Gain from m1 given family - {fi, fj}
                    is_comp_i = 1 if _is_comparable(m1, fi) else 0
                    is_comp_j = 1 if _is_comparable(m1, fj) else 0
                    gain1 = comparable_count[m1] - is_comp_i - is_comp_j
                    if gain1 <= 0:
                        continue
                    
                    for m2 in candidates:
                        if m2 == m1:
                            continue
                        # Gain from m2 given family - {fi, fj} + {m1}
                        is_comp_i2 = 1 if _is_comparable(m2, fi) else 0
                        is_comp_j2 = 1 if _is_comparable(m2, fj) else 0
                        is_comp_m1 = 1 if _is_comparable(m2, m1) else 0
                        gain2 = comparable_count[m2] - is_comp_i2 - is_comp_j2 + is_comp_m1
                        total_gain = gain1 + gain2 - total_loss
                        
                        if total_gain > best_gain:
                            best_gain = total_gain
                            best_m1 = m1
                            best_m2 = m2
                
                if best_gain > 0 and best_m1 != -1 and best_m2 != -1:
                    # Perform the 2-swap
                    family[i] = best_m1
                    family[j] = best_m2
                    family_set.discard(fi)
                    family_set.discard(fj)
                    family_set.add(best_m1)
                    family_set.add(best_m2)
                    
                    # Update comparable_count
                    for s in _SUBSETS_OF[fi]:
                        if s in family_set:
                            comparable_count[s] -= 1
                    for s in _SUPERSETS_OF[fi]:
                        if s in family_set:
                            comparable_count[s] -= 1
                    for s in _SUBSETS_OF[fj]:
                        if s in family_set:
                            comparable_count[s] -= 1
                    for s in _SUPERSETS_OF[fj]:
                        if s in family_set:
                            comparable_count[s] -= 1
                    for s in _SUBSETS_OF[best_m1]:
                        if s in family_set:
                            comparable_count[s] += 1
                    for s in _SUPERSETS_OF[best_m1]:
                        if s in family_set:
                            comparable_count[s] += 1
                    for s in _SUBSETS_OF[best_m2]:
                        if s in family_set:
                            comparable_count[s] += 1
                    for s in _SUPERSETS_OF[best_m2]:
                        if s in family_set:
                            comparable_count[s] += 1
                    
                    current_score += best_gain
                    if current_score > best_score:
                        best_score = current_score
                        best_family = list(family)
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 3:
                break
    
    return best_family, best_score

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction
    family = _greedy_construct()
    
    # Ensure we have exactly M sets
    if len(family) < M:
        family_set = set(family)
        random.seed(42)
        while len(family) < M:
            m = random.choice(_ALL_MASKS)
            if m not in family_set:
                family.append(m)
                family_set.add(m)
    
    # Phase 2: Local search (1-swap)
    elapsed = time.time() - start_time
    remaining_time = TIME_LIMIT - elapsed - 10.0  # Reserve 10s for polish + safety
    if remaining_time > 2.0:
        family, score = _local_search(family, remaining_time)
    
    # Phase 3: Polish pass (2-swap)
    elapsed = time.time() - start_time
    remaining_time = TIME_LIMIT - elapsed - 5.0  # 5s safety margin
    if remaining_time > 1.0:
        family, score = _polish(family, remaining_time)
    
    # Convert to lists of vertices
    result = []
    for m in family:
        vertices = [b for b in range(N) if (m >> b) & 1]
        result.append(vertices)
    
    return result
# EVOLVE-BLOCK-END
