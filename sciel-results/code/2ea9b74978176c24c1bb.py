"""Program for increasing sequences with unique consecutive-block sums."""
import random
import math

# EVOLVE-BLOCK-START

def _compute_block_sums_set(seq):
    """Compute all nonempty consecutive block sums for a sequence using prefix sums."""
    n = len(seq)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + seq[i]
    sums = set()
    for i in range(n):
        for j in range(i + 1, n + 1):
            sums.add(prefix[j] - prefix[i])
    return sums

def _count_duplicate_sums_fast(seq):
    """Count duplicates using prefix sums - O(n^2)."""
    n = len(seq)
    if n == 0:
        return 0
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + seq[i]
    total_blocks = n * (n + 1) // 2
    seen = set()
    dupes = 0
    for i in range(n):
        for j in range(i + 1, n + 1):
            s = prefix[j] - prefix[i]
            if s in seen:
                dupes += 1
            else:
                seen.add(s)
    return dupes

def _is_valid(seq):
    """Check if sequence has all distinct block sums and is strictly increasing in [1,256]."""
    if not seq:
        return True
    if seq[0] < 1 or seq[-1] > 256:
        return False
    for i in range(1, len(seq)):
        if seq[i] <= seq[i - 1]:
            return False
    return _count_duplicate_sums_fast(seq) == 0

def _greedy_construct():
    """Greedy construction: pick smallest valid next element."""
    values = []
    sums = set()
    for x in range(1, 257):
        tails = []
        total = x
        tails.append(total)
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _extend_greedy(seq):
    """Extend a sequence greedily by trying all possible next values."""
    if not seq:
        return [1]
    last = seq[-1]
    if last >= 256:
        return seq
    sums = _compute_block_sums_set(seq)
    for x in range(last + 1, 257):
        tails = []
        total = x
        tails.append(total)
        for a in reversed(seq):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            new_seq = seq + [x]
            sums.update(tails)
            return _extend_greedy(new_seq)
    return seq

def _sa_on_diffs(base, diffs, max_iters=20000, start_temp=5.0, end_temp=0.01, seed=42):
    """Simulated annealing on (base, diffs) representation.
    Objective: minimize duplicate block sums.
    """
    rng = random.Random(seed)
    n = len(diffs) + 1
    
    def build(base, diffs):
        s = [base]
        for d in diffs:
            s.append(s[-1] + d)
        return s
    
    def cost(base, diffs):
        seq = build(base, diffs)
        if seq[0] < 1 or seq[-1] > 256:
            return 10000
        for i in range(len(seq) - 1):
            if seq[i + 1] <= seq[i]:
                return 10000
        return _count_duplicate_sums_fast(seq)
    
    cur_cost = cost(base, diffs)
    best_cost = cur_cost
    best_base = base
    best_diffs = list(diffs)
    
    temp = start_temp
    cooling = (end_temp / start_temp) ** (1.0 / max_iters)
    
    for it in range(max_iters):
        temp *= cooling
        new_base = base
        new_diffs = list(diffs)
        move = rng.random()
        
        if move < 0.25:
            # Modify one difference
            idx = rng.randrange(len(new_diffs))
            delta = rng.randint(-4, 4)
            if delta != 0:
                new_diffs[idx] = max(1, new_diffs[idx] + delta)
        elif move < 0.40:
            # Modify base
            delta = rng.randint(-3, 3)
            if delta != 0:
                new_base = max(1, base + delta)
        elif move < 0.55:
            # Swap two differences
            if len(new_diffs) >= 2:
                i, j = rng.sample(range(len(new_diffs)), 2)
                new_diffs[i], new_diffs[j] = new_diffs[j], new_diffs[i]
        elif move < 0.70:
            # Extend by one
            last_val = new_base + sum(new_diffs)
            if last_val < 256:
                new_d = rng.randint(1, min(256 - last_val, 10))
                new_diffs.append(new_d)
        elif move < 0.80:
            # Shrink by one
            if len(new_diffs) > 1:
                new_diffs.pop()
        else:
            # Random restart of a few diffs
            for _ in range(rng.randint(1, 3)):
                idx = rng.randrange(len(new_diffs))
                new_diffs[idx] = max(1, new_diffs[idx] + rng.randint(-5, 5))
        
        new_cost = cost(new_base, new_diffs)
        delta_c = new_cost - cur_cost
        
        if delta_c <= 0 or rng.random() < math.exp(-delta_c / max(temp, 1e-9)):
            base, diffs = new_base, new_diffs
            cur_cost = new_cost
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_base = base
                best_diffs = list(diffs)
    
    return best_base, best_diffs

def _try_extend_from_valid(seq):
    """Given a valid sequence, try to extend it further."""
    if not seq or seq[-1] >= 256:
        return seq
    sums = _compute_block_sums_set(seq)
    for x in range(seq[-1] + 1, 257):
        tails = []
        total = x
        tails.append(total)
        for a in reversed(seq):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            return _try_extend_from_valid(seq + [x])
    return seq

def construct():
    """Construct a strictly increasing sequence in [1,256] with distinct consecutive block sums."""
    random.seed(42)
    
    # Phase 1: Greedy construction
    greedy = _greedy_construct()
    best_seq = _extend_greedy(list(greedy))
    best_len = len(best_seq)
    
    # Phase 2: Simulated annealing from greedy representation
    base = greedy[0]
    diffs = [greedy[i + 1] - greedy[i] for i in range(len(greedy) - 1)]
    
    for attempt in range(8):
        # Vary starting point
        if attempt == 0:
            b, d = base, list(diffs)
        elif attempt == 1:
            # Try smaller base
            b = max(1, base - 1)
            d = list(diffs)
        elif attempt == 2:
            # Perturb diffs
            b = base
            d = list(diffs)
            for i in range(len(d)):
                d[i] = max(1, d[i] + random.randint(-2, 2))
        else:
            # Random restart with similar length
            b = random.randint(1, 10)
            target_len = len(greedy) + random.randint(-2, 3)
            target_len = max(3, min(target_len, 15))
            d = [random.randint(1, max(1, (256 - b) // target_len)) for _ in range(target_len - 1)]
        
        # Run SA
        sa_base, sa_diffs = _sa_on_diffs(
            b, d,
            max_iters=15000,
            start_temp=4.0,
            end_temp=0.01,
            seed=42 + attempt
        )
        
        sa_seq = [sa_base]
        for dd in sa_diffs:
            sa_seq.append(sa_seq[-1] + dd)
        
        # Validate
        if not sa_seq or sa_seq[0] < 1 or sa_seq[-1] > 256:
            continue
        if any(sa_seq[i + 1] <= sa_seq[i] for i in range(len(sa_seq) - 1)):
            continue
        
        sa_dupes = _count_duplicate_sums_fast(sa_seq)
        
        if sa_dupes == 0:
            # Valid! Try to extend
            ext = _try_extend_from_valid(sa_seq)
            if len(ext) > best_len:
                best_seq = ext
                best_len = len(ext)
        else:
            # Not valid, but maybe we can fix it with local search
            # Try greedy extension from a valid prefix
            for cut in range(len(sa_seq) - 1, 2, -1):
                prefix = sa_seq[:cut]
                if _is_valid(prefix):
                    ext = _try_extend_from_valid(prefix)
                    if len(ext) > best_len:
                        best_seq = ext
                        best_len = len(ext)
                    break
    
    # Phase 3: Try different greedy starting points
    for start in [1, 2, 3, 4, 5, 7, 10]:
        trial = [start]
        trial = _extend_greedy(trial)
        if len(trial) > best_len:
            best_seq = trial
            best_len = len(trial)
    
    # Phase 4: Final extension
    best_seq = _try_extend_from_valid(best_seq)
    
    return best_seq

# EVOLVE-BLOCK-END
