import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_idx = {tri: i for i, tri in enumerate(all_triples)}

    # Precompute triple bitmask for each possible block (lazy cache)
    block_cache = {}
    def get_block_mask(block):
        key = tuple(sorted(block))
        if key in block_cache:
            return block_cache[key]
        mask = np.zeros(num_triples, dtype=np.int32)
        for tri in itertools.combinations(block, t):
            mask[triple_idx[tri]] += 1
        block_cache[key] = mask
        return mask

    def initial_greedy(seed=0):
        """Fast greedy to get a starting point near optimal."""
        rng = random.Random(seed)
        uncovered = np.ones(num_triples, dtype=bool)
        blocks = []
        while uncovered.any():
            unc_idx = np.where(uncovered)[0]
            start_i = unc_idx[rng.randint(0, len(unc_idx) - 1)]
            start_tri = all_triples[start_i]
            block = list(start_tri)
            block_set = set(block)
            while len(block) < k:
                best_gain = 0
                best_cands = []
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for tri in itertools.combinations(block + [x], t):
                        if x in tri and uncovered[triple_idx[tri]]:
                            # Check other t-1 elements are in block
                            others = [p for p in tri if p != x]
                            if all(p in block_set for p in others):
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_cands = [x]
                    elif gain == best_gain and gain > 0:
                        best_cands.append(x)
                if best_cands:
                    x = best_cands[rng.randint(0, len(best_cands) - 1)]
                else:
                    avail = [x for x in range(v) if x not in block_set]
                    x = avail[rng.randint(0, len(avail) - 1)]
                block.append(x)
                block_set.add(x)
            block.sort()
            blocks.append(block)
            for tri in itertools.combinations(block, t):
                uncovered[triple_idx[tri]] = False
        return blocks

    def build_coverage(blocks):
        """Return coverage count array."""
        cov = np.zeros(num_triples, dtype=np.int32)
        for b in blocks:
            mask = get_block_mask(b)
            cov += mask
        return cov

    def is_valid(cov):
        return (cov >= 1).all()

    def simulated_annealing(blocks, time_limit=8.0):
        """SA over fixed-size block list. Try to reduce block count by swapping."""
        rng = random.Random(int(time.time() * 1000) % 2**31)
        blocks = [b[:] for b in blocks]
        cov = build_coverage(blocks)
        current_uncovered = (cov < 1).sum()
        best_blocks = [b[:] for b in blocks]
        best_uncovered = current_uncovered
        best_len = len(blocks)
        
        # Try to reduce size: start from current size, allow adding/removing
        # We'll work with variable size but bias toward reduction
        n_blocks = len(blocks)
        temp = 100.0
        cool_rate = 0.995
        max_iter = 50000
        
        for it in range(max_iter):
            if time.time() > start_time + time_limit:
                break
            if temp < 0.01:
                break
                
            # Random perturbation: swap one element in a random block
            bi = rng.randint(0, n_blocks - 1)
            pos = rng.randint(0, k - 1)
            old_val = blocks[bi][pos]
            new_val = rng.randint(0, v - 1)
            if new_val == old_val:
                continue
            # Ensure no duplicate in block
            if new_val in blocks[bi] and new_val != old_val:
                continue
            
            # Update coverage
            old_mask = get_block_mask(blocks[bi])
            blocks[bi][pos] = new_val
            blocks[bi].sort()
            new_mask = get_block_mask(blocks[bi])
            
            cov -= old_mask
            cov += new_mask
            new_uncovered = (cov < 1).sum()
            
            # Accept if better or with probability exp(-delta/T)
            delta = new_uncovered - current_uncovered
            if delta <= 0 or rng.random() < np.exp(-delta / temp):
                current_uncovered = new_uncovered
                if new_uncovered < best_uncovered:
                    best_uncovered = new_uncovered
                    best_blocks = [b[:] for b in blocks]
                # Occasionally try to remove a block if coverage allows
                if new_uncovered == 0 and n_blocks > 69:
                    # Try removing a random block
                    ri = rng.randint(0, n_blocks - 1)
                    rmask = get_block_mask(blocks[ri])
                    cov_test = cov - rmask
                    if (cov_test >= 1).all():
                        cov = cov_test
                        blocks.pop(ri)
                        n_blocks -= 1
                        current_uncovered = 0
                        if n_blocks < best_len:
                            best_len = n_blocks
                            best_blocks = [b[:] for b in blocks]
            else:
                # Revert
                blocks[bi][pos] = old_val
                blocks[bi].sort()
                cov += old_mask
                cov -= new_mask
            
            temp *= cool_rate
        
        if best_uncovered == 0:
            return best_blocks
        return blocks

    def multi_start_sa(n_starts=5, time_per_start=8.0):
        """Run SA from multiple greedy starts."""
        best = None
        for s in range(n_starts):
            if time.time() > deadline - 5:
                break
            greedy_blocks = initial_greedy(seed=s * 12345 + 678)
            # Trim to reasonable size if greedy gives too many
            if len(greedy_blocks) > 80:
                # Keep only first 80 as approximation
                greedy_blocks = greedy_blocks[:80]
            sa_blocks = simulated_annealing(greedy_blocks, time_limit=time_per_start)
            if best is None or len(sa_blocks) < len(best):
                best = sa_blocks
            if len(best) <= 69:
                break
        return best

    # Main: multi-start SA
    result = multi_start_sa(n_starts=8, time_per_start=10.0)
    
    # Final validity check
    if result is not None:
        cov = build_coverage(result)
        if not is_valid(cov):
            # Fallback to greedy
            result = initial_greedy(seed=0)
    
    return result
