import time
import random

# EVOLVE-BLOCK-START

def _is_valid_full(values):
    """Check if a full sequence has all distinct consecutive block sums."""
    n = len(values)
    seen = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            if total in seen:
                return False
            seen.add(total)
    return True

def _compute_block_sums(values):
    """Compute all consecutive block sums for a sequence."""
    n = len(values)
    sums = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            sums.add(total)
    return sums

def _greedy_forward(values, max_val=256):
    """Greedy: pick smallest valid next element."""
    current = list(values)
    cur_sums = _compute_block_sums(current)
    for x in range((current[-1] if current else 0) + 1, max_val + 1):
        tails = [x]
        total = x
        for a in reversed(current):
            total += a
            tails.append(total)
        if not cur_sums.intersection(tails):
            current.append(x)
            cur_sums.update(tails)
    return current

def _max_possible_length(total_sum, max_val=256):
    """
    Upper bound on sequence length given total sum of chosen values.
    A sequence of length k has k*(k+1)/2 block sums, each in [1, total_sum].
    So k*(k+1)/2 <= total_sum, giving k <= (-1 + sqrt(1+8*total_sum))/2.
    Also k <= max_val (can't have more elements than values).
    """
    if total_sum <= 0:
        return 0
    # k*(k+1)/2 <= total_sum
    # k^2 + k - 2*total_sum <= 0
    # k <= (-1 + sqrt(1 + 8*total_sum)) / 2
    import math
    k_bound = int((-1 + math.sqrt(1 + 8 * total_sum)) / 2)
    return min(k_bound, max_val)

def _max_possible_length_with_min_val(total_sum, min_val, max_val=256):
    """
    Refined upper bound: with minimum value min_val, the minimum total sum
    for k elements is min_val + (min_val+1) + ... + (min_val+k-1)
    = k*min_val + k*(k-1)/2.
    Also need k*(k+1)/2 <= total_sum.
    """
    if total_sum <= 0:
        return 0
    import math
    # From k*(k+1)/2 <= total_sum
    k_bound1 = int((-1 + math.sqrt(1 + 8 * total_sum)) / 2)
    
    # From k*min_val + k*(k-1)/2 <= total_sum
    # k^2/2 + (min_val - 1/2)*k - total_sum <= 0
    # k <= (-a + sqrt(a^2 + 2*total_sum)) where a = 2*min_val - 1
    a = 2 * min_val - 1
    disc = a * a + 4 * total_sum
    if disc < 0:
        return 0
    k_bound2 = int((-a + math.sqrt(disc)) / 2)
    
    return min(k_bound1, k_bound2, max_val)

def _branch_and_bound_subset(max_val=256, deadline=None, best_len=0, min_val=1):
    """
    Branch and bound over subsets of {min_val, ..., max_val}.
    At each step, decide to include or exclude the current smallest available value.
    Use upper bounds to prune.
    """
    if deadline is None:
        deadline = time.time() + 50
    
    best_seq = []
    best_len_found = best_len
    
    # We'll do DFS: at each node, we have a set of chosen values (sorted),
    # and we decide whether to include the next candidate value.
    # Candidate values are processed in increasing order.
    
    # To make this tractable, we limit the depth and use bounds aggressively.
    
    def dfs(chosen, sums_set, total_sum, next_val, depth):
        nonlocal best_seq, best_len_found
        
        if time.time() > deadline:
            return
        
        k = len(chosen)
        
        # Update best
        if k > best_len_found:
            best_len_found = k
            best_seq = list(chosen)
        
        # Upper bound: how many more can we add?
        remaining_sum = 0
        # Max additional sum if we take all values from next_val to max_val
        # But we can only take values that are > last chosen
        if chosen:
            last = chosen[-1]
            start = max(next_val, last + 1)
        else:
            start = next_val
        
        if start > max_val:
            return
        
        # Max possible additional sum
        # Sum of start + (start+1) + ... + max_val
        n_avail = max_val - start + 1
        if n_avail <= 0:
            return
        max_add_sum = n_avail * (start + max_val) // 2
        
        # Upper bound on total length
        max_total_len = _max_possible_length_with_min_val(
            total_sum + max_add_sum, start if not chosen else chosen[-1] + 1, max_val
        )
        
        # Also: we can add at most n_avail more elements
        if k + min(n_avail, max_total_len) <= best_len_found:
            return
        
        # Try including next_val
        if next_val <= max_val:
            # Check if next_val can be added
            if not chosen or next_val > chosen[-1]:
                tails = [next_val]
                total = next_val
                for a in reversed(chosen):
                    total += a
                    tails.append(total)
                
                if not sums_set.intersection(tails):
                    new_sums = set(sums_set)
                    new_sums.update(tails)
                    chosen.append(next_val)
                    dfs(chosen, new_sums, total_sum + next_val, next_val + 1, depth + 1)
                    chosen.pop()
            
            # Try excluding next_val
            dfs(chosen, sums_set, total_sum, next_val + 1, depth)
    
    dfs([], set(), 0, min_val, 0)
    return best_seq, best_len_found

def _local_search_improve(seq, max_val=256, deadline=None, iterations=5000):
    """
    Local search: try to improve the sequence by swapping, removing, or adding elements.
    """
    if deadline is None:
        deadline = time.time() + 50
    
    best = list(seq)
    best_len = len(best)
    rng = random.Random(42)
    
    for _ in range(iterations):
        if time.time() > deadline:
            break
        
        if not best:
            break
        
        # Try adding a random element
        if rng.random() < 0.3:
            # Try to insert a new element
            possible_inserts = []
            for pos in range(len(best) + 1):
                lo = (best[pos-1] + 1) if pos > 0 else 1
                hi = (best[pos] - 1) if pos < len(best) else max_val
                if lo <= hi:
                    possible_inserts.append((pos, lo, hi))
            
            if possible_inserts:
                pos, lo, hi = rng.choice(possible_inserts)
                x = rng.randint(lo, hi)
                new_seq = best[:pos] + [x] + best[pos:]
                if _is_valid_full(new_seq):
                    if len(new_seq) > best_len:
                        best = new_seq
                        best_len = len(best)
        
        # Try removing a random element and re-adding
        elif len(best) > 1:
            idx = rng.randint(0, len(best) - 1)
            new_seq = best[:idx] + best[idx+1:]
            if _is_valid_full(new_seq):
                # Try to extend
                cur_sums = _compute_block_sums(new_seq)
                extended = list(new_seq)
                last = extended[-1] if extended else 0
                for x in range(last + 1, max_val + 1):
                    tails = [x]
                    total = x
                    for a in reversed(extended):
                        total += a
                        tails.append(total)
                    if not cur_sums.intersection(tails):
                        extended.append(x)
                        cur_sums.update(tails)
                        last = x
                if len(extended) > best_len:
                    best = extended
                    best_len = len(best)
        
        # Try swapping two elements
        else:
            if len(best) >= 2:
                i, j = rng.sample(range(len(best)), 2)
                if i > j:
                    i, j = j, i
                # Swap and check increasing order
                if best[i] > best[j]:  # already in order, skip
                    continue
                # Can't swap to maintain increasing order unless we re-sort
                # Actually, let's just try a different approach:
                # Remove one element and try replacing it
                idx = rng.randint(0, len(best) - 1)
                lo = (best[idx-1] + 1) if idx > 0 else 1
                hi = (best[idx+1] - 1) if idx < len(best) - 1 else max_val
                if lo < hi:
                    x = rng.randint(lo, hi)
                    if x == best[idx]:
                        continue
                    new_seq = best[:idx] + [x] + best[idx+1:]
                    if new_seq == sorted(new_seq) and _is_valid_full(new_seq):
                        if len(new_seq) > best_len:
                            best = new_seq
                            best_len = len(best)
    
    return best, best_len

def _random_prefix_greedy(rng, max_val=256, prefix_len=3):
    """Generate a random valid prefix, then greedily extend."""
    prefix = []
    cur_sums = set()
    last = 0
    
    for _ in range(prefix_len):
        found = False
        for _ in range(200):
            x = rng.randint(last + 1, max_val)
            tails = [x]
            total = x
            for a in reversed(prefix):
                total += a
                tails.append(total)
            if not cur_sums.intersection(tails):
                prefix.append(x)
                cur_sums.update(tails)
                last = x
                found = True
                break
        if not found:
            break
    
    if not prefix:
        return []
    
    # Greedily extend
    seq = list(prefix)
    cur_sums = _compute_block_sums(seq)
    last = seq[-1]
    for x in range(last + 1, max_val + 1):
        tails = [x]
        total = x
        for a in reversed(seq):
            total += a
            tails.append(total)
        if not cur_sums.intersection(tails):
            seq.append(x)
            cur_sums.update(tails)
    
    return seq

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill
    
    best = []
    best_len = 0
    
    # 1. Greedy initial solution
    greedy_seq = _greedy_forward([])
    if len(greedy_seq) > best_len:
        best = greedy_seq
        best_len = len(best)
    
    # 2. Branch and bound from different starting points
    if time.time() < deadline - 30:
        # B&B starting from value 1
        bb_seq, bb_len = _branch_and_bound_subset(
            max_val=256, deadline=min(time.time() + 15, deadline - 15),
            best_len=best_len, min_val=1
        )
        if bb_len > best_len:
            best = bb_seq
            best_len = best_len = bb_len
    
    # 3. Local search improvement
    if time.time() < deadline - 10:
        ls_seq, ls_len = _local_search_improve(
            best, max_val=256, deadline=min(time.time() + 10, deadline - 5)
        )
        if ls_len > best_len:
            best = ls_seq
            best_len = ls_len
    
    # 4. Randomized restarts with different strategies
    if time.time() < deadline - 5:
        rng = random.Random(99999)
        
        for restart in range(20):
            if time.time() > deadline - 2:
                break
            
            # Strategy A: random prefix + greedy
            prefix_seq = _random_prefix_greedy(rng, max_val=256, prefix_len=rng.randint(1, 4))
            if len(prefix_seq) > best_len:
                best = prefix_seq
                best_len = len(prefix_seq)
            
            # Strategy B: B&B with different min_val
            if time.time() < deadline - 5:
                min_v = rng.randint(1, 20)
                bb_seq2, bb_len2 = _branch_and_bound_subset(
                    max_val=256, deadline=min(time.time() + 3, deadline - 3),
                    best_len=best_len, min_val=min_v
                )
                if bb_len2 > best_len:
                    best = bb_seq2
                    best_len = bb_len2
    
    # 5. Final local search polish
    if time.time() < deadline - 2 and best:
        ls_seq2, ls_len2 = _local_search_improve(
            best, max_val=256, deadline=min(time.time() + 3, deadline - 1)
        )
        if ls_len2 > best_len:
            best = ls_seq2
            best_len = ls_len2
    
    # Final validation
    if not best or not _is_valid_full(best):
        best = _greedy_forward([])
    
    return best

# EVOLVE-BLOCK-END
