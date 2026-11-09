import time
import itertools
import numpy as np
from collections import defaultdict

_SEED = 20240612

def _rng(seed):
    return np.random.RandomState(seed)

def _compute_block_triple_indices(block, t, triple_to_idx):
    """Return the list of triple indices covered by this block."""
    return [triple_to_idx[tr] for tr in itertools.combinations(block, t)]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    rng = _rng(_SEED)

    # Precompute all triples and map to indices
    all_triples = list(itertools.combinations(range(v), t))
    n_all = len(all_triples)
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}

    # Precompute direct pair->(third_point)->triple_idx mapping for O(1) lookup
    pair_third_to_triple = {}
    for tr_idx, tr in enumerate(all_triples):
        for i in range(t):
            for j in range(i+1, t):
                for p in tr:
                    if p != tr[i] and p != tr[j]:
                        c = p
                        break
                a, b = tr[i], tr[j]
                if a > b:
                    a, b = b, a
                pair_third_to_triple[(a, b, c)] = tr_idx

    # For each point, precompute which pairs it forms with other points
    point_pairs = defaultdict(list)
    for a in range(v):
        for b in range(a+1, v):
            point_pairs[a].append((b, (a, b)))
            point_pairs[b].append((a, (a, b)))

    # Count of how many times each triple is covered
    count = np.zeros(n_all, dtype=np.int32)
    blocks = []
    block_set = set()
    block_triple_idx_list = []  # parallel to blocks: list of triple indices for each block

    def add_block(block):
        if block in block_set:
            return False
        block_set.add(block)
        blocks.append(block)
        t_idx_list = _compute_block_triple_indices(block, t, triple_to_idx)
        block_triple_idx_list.append(t_idx_list)
        for idx in t_idx_list:
            count[idx] += 1
        return True

    def remove_block(block, block_idx=None):
        if block_idx is None:
            block_idx = blocks.index(block)
        for idx in block_triple_idx_list[block_idx]:
            count[idx] -= 1
        blocks.pop(block_idx)
        block_triple_idx_list.pop(block_idx)
        block_set.discard(block)

    def get_uncovered_indices():
        return np.where(count == 0)[0]

    def build_block_greedy(start_triple, count_arr, rng_obj):
        """Build a block greedily starting from a given triple. Optimized with vectorized scoring."""
        block = list(start_triple)
        remaining = set(range(v))
        for p in start_triple:
            remaining.discard(p)

        for _ in range(k - t):
            best_cand = None
            best_s = -1
            pairs = list(itertools.combinations(block, t-1))
            
            # Vectorized score computation
            cand_list = list(remaining)
            if not cand_list:
                break
            
            # For each candidate, compute how many uncovered triples it would cover
            scores = np.zeros(len(cand_list), dtype=np.int32)
            for ci, cand in enumerate(cand_list):
                s = 0
                for p1, p2 in pairs:
                    a, b = (p1, p2) if p1 < p2 else (p2, p1)
                    tr_idx = pair_third_to_triple.get((a, b, cand))
                    if tr_idx is not None and count_arr[tr_idx] == 0:
                        s += 1
                scores[ci] = s
            
            best_idx = int(np.argmax(scores))
            best_cand = cand_list[best_idx]
            
            block.append(best_cand)
            remaining.discard(best_cand)
        
        return tuple(sorted(block))

    def compute_gain(block, count_arr):
        """Compute how many uncovered triples this block covers."""
        gain = 0
        for tr in itertools.combinations(block, t):
            if count_arr[triple_to_idx[tr]] == 0:
                gain += 1
        return gain

    def compute_overlap_score(block, count_arr):
        """Compute a weighted score: gain + small bonus for low-redundancy triples."""
        gain = 0
        redundancy_penalty = 0.0
        for tr in itertools.combinations(block, t):
            idx = triple_to_idx[tr]
            if count_arr[idx] == 0:
                gain += 1
            else:
                redundancy_penalty += 0.1 * count_arr[idx]
        return gain - redundancy_penalty

    def build_block_probabilistic(start_triple, count_arr, rng_obj, alpha=0.3):
        """Build a block using probabilistic selection with softmax over candidate scores."""
        block = list(start_triple)
        remaining = set(range(v))
        for p in start_triple:
            remaining.discard(p)

        for _ in range(k - t):
            if not remaining:
                break
            
            pairs = list(itertools.combinations(block, t-1))
            
            cand_list = list(remaining)
            if not cand_list:
                break
            
            # Compute scores for all candidates
            scores = np.zeros(len(cand_list), dtype=np.float64)
            for ci, cand in enumerate(cand_list):
                s = 0
                for p1, p2 in pairs:
                    a, b = (p1, p2) if p1 < p2 else (p2, p1)
                    tr_idx = pair_third_to_triple.get((a, b, cand))
                    if tr_idx is not None and count_arr[tr_idx] == 0:
                        s += 1
                scores[ci] = s
            
            if alpha > 0 and len(scores) > 1:
                max_score = np.max(scores)
                exp_scores = np.exp((scores - max_score) / alpha)
                total = np.sum(exp_scores)
                if total > 0:
                    probs = exp_scores / total
                    best_idx = int(rng_obj.choice(len(cand_list), p=probs))
                else:
                    best_idx = int(np.argmax(scores))
            else:
                best_idx = int(np.argmax(scores))
            
            best_cand = cand_list[best_idx]
            block.append(best_cand)
            remaining.discard(best_cand)
        
        return tuple(sorted(block))

    def find_replacement_block(critical_triples, count_arr, rng_obj):
        """Find a replacement block that covers all critical triples using targeted search."""
        if not critical_triples:
            return None
        
        best_block = None
        best_score = -1e9
        n_crit = len(critical_triples)
        
        # Strategy 1: Try building from each critical triple with varying alpha
        alphas = [0.0, 0.15, 0.3, 0.5]
        for start_tr in critical_triples[:8]:
            for alpha in alphas:
                block = build_block_probabilistic(start_tr, count_arr, rng_obj, alpha)
                block_triple_set = set(itertools.combinations(block, t))
                covered = sum(1 for tr in critical_triples if tr in block_triple_set)
                if covered == n_crit:
                    gain = compute_gain(block, count_arr)
                    score = gain
                    if score > best_score:
                        best_score = score
                        best_block = block
        
        # Strategy 2: Anchor on high-frequency points in critical triples
        point_freq = defaultdict(int)
        for tr in critical_triples:
            for p in tr:
                point_freq[p] += 1
        sorted_points = sorted(point_freq.keys(), key=lambda p: -point_freq[p])
        
        for p in sorted_points[:6]:
            relevant = [tr for tr in critical_triples if p in tr]
            if not relevant:
                continue
            for start_tr in relevant[:3]:
                for alpha in [0.0, 0.2, 0.4]:
                    block = build_block_probabilistic(start_tr, count_arr, rng_obj, alpha)
                    block_triple_set = set(itertools.combinations(block, t))
                    covered = sum(1 for tr in critical_triples if tr in block_triple_set)
                    if covered == n_crit:
                        gain = compute_gain(block, count_arr)
                        if gain > best_score:
                            best_score = gain
                            best_block = block
        
        return best_block

    # Phase 1: Greedy construction with best-first selection
    # Instead of random sampling, always pick the uncovered triple that leads to
    # the block with maximum gain (most uncovered triples covered).
    max_phase1_iters = 500
    for _iter in range(max_phase1_iters):
        if time.time() > deadline - 15:
            break
        
        uncovered_indices = get_uncovered_indices()
        if len(uncovered_indices) == 0:
            break

        # Sample a diverse set of uncovered triples to evaluate
        sample_size = min(40, len(uncovered_indices))
        sample_idx = rng.choice(uncovered_indices, size=sample_size, replace=False)

        best_block = None
        best_score = -1e9

        for idx in sample_idx:
            tr = all_triples[idx]
            # Primarily use greedy (alpha=0) for speed, occasionally probabilistic
            alphas_to_try = [0.0]
            if _iter % 3 == 0:
                alphas_to_try = [0.0, 0.2, 0.4]
            for alpha in alphas_to_try:
                block = build_block_probabilistic(tr, count, rng, alpha)
                score = compute_overlap_score(block, count)
                if score > best_score:
                    best_score = score
                    best_block = block

        if best_block is None:
            break

        add_block(best_block)

    # Phase 2: Local improvement - try removing or replacing blocks
    # Use exact block_triple_idx_list for fast critical triple detection
    for _round in range(200):
        if time.time() > deadline - 5:
            break
        improved = False
        n_b = len(blocks)
        for bi in range(n_b):
            if time.time() > deadline - 5:
                break
            
            removed_block = blocks[bi]
            # Identify critical triples (count==1) using precomputed index list
            critical_triples = []
            for tr_idx in block_triple_idx_list[bi]:
                if count[tr_idx] == 1:
                    critical_triples.append(all_triples[tr_idx])
            
            if not critical_triples:
                # Can remove this block safely
                remove_block(removed_block, bi)
                improved = True
                break
            
            # Try to find a replacement block that covers all critical triples
            best_repl = find_replacement_block(critical_triples, count, rng)
            
            if best_repl is not None and best_repl != removed_block:
                # Remove old, add new
                remove_block(removed_block, bi)
                add_block(best_repl)
                improved = True
                break

        if not improved:
            break

    # Phase 3: Ensure full coverage
    while time.time() < deadline - 2:
        uncovered_indices = get_uncovered_indices()
        if len(uncovered_indices) == 0:
            break
        # Pick the uncovered triple that is in the fewest potential blocks (hardest)
        # For now, just take the first one
        tr = all_triples[uncovered_indices[0]]
        block = build_block_probabilistic(tr, count, rng, 0.0)
        add_block(block)

    # Phase 4: Final polish - try to remove redundant blocks (using precomputed indices)
    for _round in range(100):
        if time.time() > deadline - 1:
            break
        improved = False
        n_b = len(blocks)
        for bi in range(n_b):
            if time.time() > deadline - 1:
                break
            
            can_remove = True
            for tr_idx in block_triple_idx_list[bi]:
                if count[tr_idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                remove_block(blocks[bi], bi)
                improved = True
                break
        
        if not improved:
            break

    # Phase 5: 2-for-1 replacement - try replacing two blocks with one better block
    for _round in range(60):
        if time.time() > deadline - 1:
            break
        improved = False
        if len(blocks) < 2:
            break
        n_blocks = len(blocks)
        for _attempt in range(15):
            if time.time() > deadline - 1:
                break
            i, j = rng.choice(n_blocks, size=2, replace=False)
            if i > j:
                i, j = j, i
            
            block_i = blocks[i]
            block_j = blocks[j]
            
            # Find critical triples using precomputed index lists
            critical_triples = []
            seen_crit = set()
            for tr_idx in block_triple_idx_list[i]:
                if count[tr_idx] == 1:
                    tr = all_triples[tr_idx]
                    if tr not in seen_crit:
                        critical_triples.append(tr)
                        seen_crit.add(tr)
            for tr_idx in block_triple_idx_list[j]:
                if count[tr_idx] == 1:
                    tr = all_triples[tr_idx]
                    if tr not in seen_crit:
                        critical_triples.append(tr)
                        seen_crit.add(tr)
            
            if not critical_triples:
                # Both blocks can be removed safely
                remove_block(block_j, j)
                remove_block(block_i, i)
                improved = True
                break
            
            # Try to build a single block that covers all critical triples
            point_freq = defaultdict(int)
            for tr in critical_triples:
                for p in tr:
                    point_freq[p] += 1
            sorted_points = sorted(point_freq.keys(), key=lambda p: -point_freq[p])
            
            best_block = None
            best_gain = -1
            
            for p in sorted_points[:4]:
                relevant = [tr for tr in critical_triples if p in tr]
                if not relevant:
                    continue
                for start_tr in relevant[:2]:
                    for alpha in [0.0, 0.2]:
                        block = build_block_probabilistic(start_tr, count, rng, alpha)
                        block_triple_set = set(itertools.combinations(block, t))
                        if all(tr in block_triple_set for tr in critical_triples):
                            gain = compute_gain(block, count)
                            if gain > best_gain:
                                best_gain = gain
                                best_block = block
            
            if best_block is not None:
                remove_block(block_j, j)
                remove_block(block_i, i)
                add_block(best_block)
                improved = True
                break
        
        if not improved:
            break

    # Final deduplication (should be empty due to block_set, but safety check)
    seen = set()
    unique_blocks = []
    for b in blocks:
        if b not in seen:
            seen.add(b)
            unique_blocks.append(b)

    return [list(b) for b in unique_blocks]
