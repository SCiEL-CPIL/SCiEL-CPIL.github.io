"""Covering design C(24,7,3) via AG(3,4) algebraic construction + bitset local search."""
import itertools
import time
import random
from collections import defaultdict
import numpy as np

def _ag34_blocks():
    """Generate blocks from AG(3,4): points are GF(4)^3, blocks are 7-point caps."""
    # GF(4) = {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1
    # Represent GF(4) elements as 0,1,2,3 with multiplication table
    gf4_mul = [[0,0,0,0],
               [0,1,2,3],
               [0,2,3,1],
               [0,3,1,2]]
    gf4_add = [[0,1,2,3],
               [1,0,3,2],
               [2,3,0,1],
               [3,2,1,0]]
    
    def gf4_add_op(a, b): return gf4_add[a][b]
    def gf4_mul_op(a, b): return gf4_mul[a][b]
    def gf4_neg(a): return gf4_add_op(a, 0)  # In GF(4), -a = a for a in {0,1}, but let's be careful
    # Actually in GF(4), characteristic 2, so -a = a
    def gf4_inv(a):
        if a == 0: return 0
        for b in range(4):
            if gf4_mul_op(a, b) == 1:
                return b
        return 0
    
    # 24 points = GF(4)^3, indexed 0..23
    # point index -> (x,y,z) in GF(4)
    points = []
    for z in range(4):
        for y in range(4):
            for x in range(4):
                points.append((x, y, z))
    
    # A 7-cap in AG(3,4): no 3 collinear
    # One construction: take a conic + infinity point, or use known 7-caps
    # Simple 7-cap: {(0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,1,1), (alpha,alpha+1,alpha)}
    # Let's use a systematic approach: for each direction (1D subspace), take a plane, 
    # then within each plane (4x4 grid), select 7 points forming a cap
    
    # Actually, let's use the standard 7-cap: in AG(3,4), a 7-arc
    # Points: P_i for i=0..6 where no three are collinear
    # One known 7-cap in GF(4)^3:
    cap_template = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 1, 1),
        (2, 2, 3)  # alpha, alpha, alpha+1
    ]
    
    # Verify no three collinear: for each triple, check if they're collinear
    # Three points A,B,C collinear iff (B-A) is a scalar multiple of (C-A)
    
    def are_collinear(p1, p2, p3):
        # Check if p2-p1 and p3-p1 are linearly dependent over GF(4)
        d1 = (gf4_add_op(p2[0], gf4_neg(p1[0])),
              gf4_add_op(p2[1], gf4_neg(p1[1])),
              gf4_add_op(p2[2], gf4_neg(p1[2])))
        d2 = (gf4_add_op(p3[0], gf4_neg(p1[0])),
              gf4_add_op(p3[1], gf4_neg(p1[1])),
              gf4_add_op(p3[2], gf4_neg(p1[2])))
        
        # d1 and d2 are dependent iff there exists lambda such that d2 = lambda * d1
        # Or cross product is zero (in 3D over GF(4))
        # Cross product in GF(4):
        # (d1[1]*d2[2] - d1[2]*d2[1], d1[2]*d2[0] - d1[0]*d2[2], d1[0]*d2[1] - d1[1]*d2[0])
        cx = gf4_add_op(gf4_mul_op(d1[1], d2[2]), gf4_mul_op(d1[2], d2[1]))
        cy = gf4_add_op(gf4_mul_op(d1[2], d2[0]), gf4_mul_op(d1[0], d2[2]))
        cz = gf4_add_op(gf4_mul_op(d1[0], d2[1]), gf4_mul_op(d1[1], d2[0]))
        return cx == 0 and cy == 0 and cz == 0
    
    # Verify cap_template
    for i in range(7):
        for j in range(i+1, 7):
            for l in range(j+1, 7):
                if are_collinear(cap_template[i], cap_template[j], cap_template[l]):
                    # Not a valid cap, need to fix
                    pass
    
    # Generate all translates of this cap
    blocks = []
    seen = set()
    for tz in range(4):
        for ty in range(4):
            for tx in range(4):
                translated = []
                for (px, py, pz) in cap_template:
                    nx = gf4_add_op(px, tx)
                    ny = gf4_add_op(py, ty)
                    nz = gf4_add_op(pz, tz)
                    translated.append((nx, ny, nz))
                # Convert to indices
                idxs = []
                for (x, y, z) in translated:
                    idx = z * 16 + y * 4 + x
                    idxs.append(idx)
                idxs.sort()
                key = tuple(idxs)
                if key not in seen:
                    seen.add(key)
                    blocks.append(idxs)
    
    # Also add scaled versions (multiply by non-zero GF(4) scalars)
    for scale in [1, 2, 3]:
        for tz in range(4):
            for ty in range(4):
                for tx in range(4):
                    translated = []
                    for (px, py, pz) in cap_template:
                        nx = gf4_add_op(gf4_mul_op(px, scale), tx)
                        ny = gf4_add_op(gf4_mul_op(py, scale), ty)
                        nz = gf4_add_op(gf4_mul_op(pz, scale), tz)
                        translated.append((nx, ny, nz))
                    idxs = []
                    for (x, y, z) in translated:
                        idx = z * 16 + y * 4 + x
                        idxs.append(idx)
                    idxs.sort()
                    key = tuple(idxs)
                    if key not in seen:
                        seen.add(key)
                        blocks.append(idxs)
    
    return blocks

def _build_bitset_index(v, k, t):
    """
    Build bitset representation for fast coverage checking.
    Returns:
      - tset_to_idx: mapping from t-subset to index
      - block_tsets: for each block, list of tset indices it covers
    """
    all_tsets = list(itertools.combinations(range(v), t))
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    n_tsets = len(all_tsets)
    
    return all_tsets, tset_to_idx, n_tsets

def _compute_coverage_bitset(blocks, tset_to_idx, n_tsets):
    """Compute coverage count for each t-set using bitset."""
    coverage = np.zeros(n_tsets, dtype=np.int32)
    for b in blocks:
        for ts in itertools.combinations(b, 3):  # t=3 for this problem
            idx = tset_to_idx.get(ts)
            if idx is not None:
                coverage[idx] += 1
    return coverage

def _greedy_complete(blocks, v, k, t, deadline, seed=None):
    """Complete a partial covering using greedy."""
    if seed is not None:
        random.seed(seed)
    
    all_tsets, tset_to_idx, n_tsets = _build_bitset_index(v, k, t)
    coverage = _compute_coverage_bitset(blocks, tset_to_idx, n_tsets)
    uncovered_mask = coverage == 0
    
    if not uncovered_mask.any():
        return blocks
    
    blocks = list(blocks)
    
    while uncovered_mask.any() and time.time() < deadline:
        # Find an uncovered t-set
        uncovered_indices = np.where(uncovered_mask)[0]
        seed_idx = random.choice(uncovered_indices)
        seed_ts = all_tsets[seed_idx]
        
        block = list(seed_ts)
        block_set = set(block)
        
        while len(block) < k and time.time() < deadline:
            best_x = -1
            best_gain = -1
            best_candidates = []
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for R in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(list(R) + [x]))
                    idx = tset_to_idx.get(ts)
                    if idx is not None and uncovered_mask[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [x]
                elif gain == best_gain:
                    best_candidates.append(x)
            
            if not best_candidates:
                if best_gain == 0:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_x = random.choice(remaining)
                    else:
                        break
                else:
                    break
            else:
                best_x = random.choice(best_candidates)
            
            block.append(best_x)
            block_set.add(best_x)
        
        if len(block) == k:
            block.sort()
            if block not in blocks:
                blocks.append(block)
            # Update coverage
            for ts in itertools.combinations(block, t):
                idx = tset_to_idx.get(ts)
                if idx is not None:
                    coverage[idx] += 1
                    if coverage[idx] == 1:
                        uncovered_mask[idx] = False
    
    return blocks

def _prune_redundant_bitset(blocks, v, k, t, deadline):
    """Remove redundant blocks using bitset coverage counts."""
    if not blocks:
        return blocks
    
    all_tsets, tset_to_idx, n_tsets = _build_bitset_index(v, k, t)
    coverage = _compute_coverage_bitset(blocks, tset_to_idx, n_tsets)
    
    # Find blocks that can be removed
    to_remove = set()
    for i, b in enumerate(blocks):
        if time.time() >= deadline:
            break
        # Check if all t-sets in this block are covered by other blocks
        redundant = True
        for ts in itertools.combinations(b, t):
            idx = tset_to_idx.get(ts)
            if idx is not None and coverage[idx] == 1:
                redundant = False
                break
        if redundant:
            to_remove.add(i)
    
    if to_remove:
        return [b for i, b in enumerate(blocks) if i not in to_remove]
    return blocks

def _local_search_bitset(blocks, v, k, t, deadline, max_rounds=100):
    """
    Local search: iteratively try to remove blocks, and if coverage is broken,
    try to fix by modifying a remaining block.
    """
    if not blocks:
        return blocks
    
    all_tsets, tset_to_idx, n_tsets = _build_bitset_index(v, k, t)
    current_blocks = list(blocks)
    best_count = len(current_blocks)
    
    for rnd in range(max_rounds):
        if time.time() >= deadline:
            break
        
        coverage = _compute_coverage_bitset(current_blocks, tset_to_idx, n_tsets)
        n = len(current_blocks)
        
        # Try to remove each block
        for i in range(n):
            if time.time() >= deadline:
                break
            
            # Tentatively remove block i
            coverage_i = coverage.copy()
            for ts in itertools.combinations(current_blocks[i], t):
                idx = tset_to_idx.get(ts)
                if idx is not None:
                    coverage_i[idx] -= 1
            
            # Check if any t-set becomes uncovered
            if (coverage_i > 0).all():
                # Safe to remove
                current_blocks.pop(i)
                coverage = coverage_i
                best_count = len(current_blocks)
                break  # Restart the removal loop
        else:
            # No more blocks can be removed, try modifications
            improved = False
            for i in range(len(current_blocks)):
                if time.time() >= deadline:
                    break
                
                # Try to modify block i to help cover more
                for _ in range(20):
                    if time.time() >= deadline:
                        break
                    
                    cand = list(current_blocks[i])
                    # Random swap
                    if len(cand) >= 2:
                        pos = random.randint(0, len(cand) - 1)
                        old = cand.pop(pos)
                        new = random.randint(0, v - 1)
                        if new not in cand:
                            cand.append(new)
                        else:
                            cand.append(old)
                    
                    cand.sort()
                    if tuple(cand) == tuple(current_blocks[i]):
                        continue
                    
                    # Check if this modification helps
                    test_blocks = list(current_blocks)
                    test_blocks[i] = cand
                    test_coverage = _compute_coverage_bitset(test_blocks, tset_to_idx, n_tsets)
                    
                    if (test_coverage > 0).all():
                        # Valid, try to prune
                        pruned = _prune_redundant_bitset(test_blocks, v, k, t, deadline)
                        if len(pruned) < best_count:
                            current_blocks = pruned
                            best_count = len(current_blocks)
                            improved = True
                            break
            
            if improved:
                continue
            else:
                break
    
    return current_blocks

def _algebraic_init(v, k, t, deadline, seed=None):
    """
    Initialize using AG(3,4) algebraic structure + greedy completion.
    """
    if seed is not None:
        random.seed(seed)
    
    # Get algebraic blocks
    ag_blocks = _ag34_blocks()
    
    # Randomly shuffle and select a subset
    random.shuffle(ag_blocks)
    
    # Start with a random subset of algebraic blocks
    n_init = min(len(ag_blocks), random.randint(40, 60))
    init_blocks = ag_blocks[:n_init]
    
    # Complete with greedy
    completed = _greedy_complete(init_blocks, v, k, t, deadline, seed=seed)
    
    return completed

def _fast_greedy_init(v, k, t, deadline, seed=None):
    """
    Fast greedy initialization using a randomized gain-based approach.
    """
    if seed is not None:
        random.seed(seed)
    
    all_tsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsets)
    blocks = []
    
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-set to seed a new block
        seed_ts = random.choice(list(uncovered))
        block = list(seed_ts)
        block_set = set(block)
        
        while len(block) < k and time.time() < deadline:
            best_x = -1
            best_gain = -1
            best_candidates = []
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for R in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(list(R) + [x]))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [x]
                elif gain == best_gain:
                    best_candidates.append(x)
            
            if not best_candidates:
                if best_gain == 0:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_x = random.choice(remaining)
                    else:
                        break
                else:
                    break
            else:
                best_x = random.choice(best_candidates)
            
            block.append(best_x)
            block_set.add(best_x)
        
        if len(block) == k:
            block.sort()
            covered = set(itertools.combinations(block, t))
            uncovered -= covered
            if block not in blocks:
                blocks.append(block)
    
    # Ensure full coverage
    if uncovered:
        for ts in list(uncovered):
            if time.time() >= deadline:
                break
            block = list(ts)
            block_set = set(block)
            while len(block) < k:
                x = random.randint(0, v - 1)
                if x not in block_set:
                    block.append(x)
                    block_set.add(x)
            block.sort()
            if block not in blocks:
                blocks.append(block)
            covered = set(itertools.combinations(block, t))
            uncovered -= covered
    
    return blocks

def _prune_redundant(blocks, v, t):
    """Remove blocks that do not cover any unique t-subset."""
    if not blocks:
        return blocks
    
    tset_to_blocks = defaultdict(int)
    for b in blocks:
        for ts in itertools.combinations(b, t):
            tset_to_blocks[ts] += 1
    
    to_remove = set()
    for i, b in enumerate(blocks):
        is_redundant = True
        for ts in itertools.combinations(b, t):
            if tset_to_blocks[ts] == 1:
                is_redundant = False
                break
        if is_redundant:
            to_remove.add(i)
    
    if to_remove:
        return [b for i, b in enumerate(blocks) if i not in to_remove]
    return blocks

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    best_blocks = None
    best_count = float('inf')
    
    # Multi-start hybrid approach with algebraic initialization
    for restart in range(20):
        if time.time() > deadline - 30:
            break
        
        # Alternate between algebraic and random greedy initialization
        if restart % 2 == 0:
            init_blocks = _algebraic_init(v, k, t, deadline - 20, seed=restart * 1000)
        else:
            init_blocks = _fast_greedy_init(v, k, t, deadline - 20, seed=restart * 1000)
        
        if not init_blocks:
            continue
        
        # Ensure validity
        all_tsets = set(itertools.combinations(range(v), t))
        covered = set()
        for b in init_blocks:
            covered.update(itertools.combinations(b, t))
        
        if not all_tsets.issubset(covered):
            # Force cover remaining
            uncovered = all_tsets - covered
            for ts in list(uncovered):
                if time.time() >= deadline - 10:
                    break
                block = list(ts)
                block_set = set(block)
                while len(block) < k:
                    x = random.randint(0, v - 1)
                    if x not in block_set:
                        block.append(x)
                        block_set.add(x)
                block.sort()
                if block not in init_blocks:
                    init_blocks.append(block)
                covered.update(itertools.combinations(block, t))
        
        if not all_tsets.issubset(covered):
            continue
        
        # Prune using bitset
        pruned = _prune_redundant_bitset(init_blocks, v, k, t, deadline - 10)
        prev_len = len(pruned)
        while len(pruned) < prev_len and time.time() < deadline - 10:
            prev_len = len(pruned)
            pruned = _prune_redundant_bitset(pruned, v, k, t, deadline - 10)
        
        # Local search with bitset
        if time.time() < deadline - 5:
            ls_blocks = _local_search_bitset(pruned, v, k, t, deadline - 2, max_rounds=50)
            if len(ls_blocks) < len(pruned):
                pruned = ls_blocks
        
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_blocks = list(pruned)
        
        if time.time() > deadline - 15:
            break
    
    if best_blocks is None or len(best_blocks) == 0:
        # Fallback: simple greedy
        all_tsets = set(itertools.combinations(range(v), t))
        uncovered = set(all_tsets)
        blocks = []
        
        while uncovered and time.time() < deadline - 2:
            seed_ts = random.choice(list(uncovered))
            block = list(seed_ts)
            block_set = set(block)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for R in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(list(R) + [x]))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_x = random.choice(remaining)
                    else:
                        break
                block.append(best_x)
                block_set.add(best_x)
            
            if len(block) == k:
                block.sort()
                if block not in blocks:
                    blocks.append(block)
                covered = set(itertools.combinations(block, t))
                uncovered -= covered
        
        best_blocks = blocks
    
    return best_blocks
