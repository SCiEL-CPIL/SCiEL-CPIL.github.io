"""Size-efficient sorting network on 16 wires."""
N = 16

# EVOLVE-BLOCK-START
def _verify(net, N):
    """Verify a network sorts all 2^N Boolean inputs (zero-one principle)."""
    for mask in range(1 << N):
        inputs = [(mask >> i) & 1 for i in range(N)]
        for (i, j) in net:
            if inputs[i] > inputs[j]:
                inputs[i], inputs[j] = inputs[j], inputs[i]
        for k in range(N - 1):
            if inputs[k] > inputs[k + 1]:
                return False
    return True

def _dedup_and_minimize(net, N):
    """Remove redundant comparators while preserving correctness."""
    if not net:
        return net
    # First verify the whole network
    if not _verify(net, N):
        return net
    # Try removing each comparator in reverse order
    result = list(net)
    for idx in range(len(result) - 1, -1, -1):
        candidate = result[:idx] + result[idx+1:]
        if _verify(candidate, N):
            result = candidate
    return result

def _aks_butterfly_layer(n, level):
    """
    Generate one butterfly layer for AKS-like network.
    For n wires at a given level, the butterfly structure connects
    wire i with wire i XOR (n >> level) for appropriate i.
    """
    n_half = n >> (level + 1)
    stride = n >> level
    comparators = []
    for block_start in range(0, n, 2 * stride):
        for i in range(stride):
            a = block_start + i
            b = block_start + stride + i
            if a < b:
                comparators.append((a, b))
    return comparators

def _build_ak_like_network(N):
    """
    Build a sorting network for N wires using a butterfly/balanced structure.
    For N=16, we use a depth-4 butterfly with additional sorting stages.
    """
    import math
    
    comparators = []
    log_n = int(math.log2(N))
    
    # Stage 1: Butterfly layers (log_n stages)
    # Each stage connects wires that differ by 2^(log_n - stage)
    for level in range(log_n - 1, -1, -1):
        layer = _aks_butterfly_layer(N, level)
        comparators.extend(layer)
    
    # Stage 2: Additional sorting to ensure complete order
    # Add bitonic-style merge stages on top of butterfly
    # First, do a full bitonic merge on the whole set
    def bitonic_merge(n, offset=0):
        if n <= 1:
            return []
        h = n // 2
        comps = []
        # Compare i with i+h for i in 0..h-1
        for i in range(h):
            comps.append((offset + i, offset + i + h))
        # Recursively merge halves
        comps.extend(bitonic_merge(h, offset))
        comps.extend(bitonic_merge(h, offset + h))
        return comps
    
    comparators.extend(bitonic_merge(N))
    
    return comparators

def _build_known_good_16():
    """
    Return a known good sorting network for 16 wires.
    We use a construction based on Batcher's odd-even merge sort,
    which for N=16 gives a specific number of comparators.
    """
    def oom(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        h = n // 2
        left = oom(h)
        right = oom(h)
        right = [(i + h, j + h) for (i, j) in right]
        base = left + right
        merge = odd_even_merge(n)
        return base + merge
    
    def odd_even_merge(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        h = n // 2
        comps = []
        # Odd merge: compare (2k+1, 2k+1+h) for k=0..h/2-1
        for k in range(h // 2):
            i = 2 * k + 1
            if i < n:
                comps.append((i, i + h))
        # Even merge: compare (2k, 2k+h) for k=0..h/2-1
        for k in range(h // 2):
            i = 2 * k
            if i < n:
                comps.append((i, i + h))
        # Recursively merge odd and even subsequences
        # Odd subsequence: indices 1, 3, 5, ..., 2h-1 -> remap to 0..h-1
        odd_sub = odd_even_merge(h)
        for (a, b) in odd_sub:
            comps.append((2 * a + 1, 2 * b + 1))
        # Even subsequence: indices 0, 2, 4, ..., 2h-2 -> remap to 0..h-1
        even_sub = odd_even_merge(h)
        for (a, b) in even_sub:
            comps.append((2 * a, 2 * b))
        return comps
    
    return oom(N)

def construct():
    """
    Construct a fixed compare-exchange network that sorts all 16-input sequences
    using as few comparators as possible.
    Returns an ordered list of integer pairs (i, j), 0 <= i < j < N.
    """
    # Try multiple constructions and pick the best verified one
    candidates = []
    
    # 1. AKS-like butterfly + bitonic merge
    try:
        net1 = _build_ak_like_network(N)
        candidates.append(net1)
    except Exception:
        pass
    
    # 2. Known good Batcher OOM
    try:
        net2 = _build_known_good_16()
        candidates.append(net2)
    except Exception:
        pass
    
    # 3. Bitonic sort network
    def bitonic_sort(n, offset=0, ascending=True):
        if n == 1:
            return []
        h = n // 2
        left = bitonic_sort(h, offset, ascending)
        right = bitonic_sort(h, offset + h, not ascending)
        comps = left + right
        # Bitonic merge
        def bitonic_merge_sub(nn, off):
            if nn <= 1:
                return []
            hh = nn // 2
            c = []
            for i in range(hh):
                a = off + i
                b = off + i + hh
                if a < b:
                    c.append((a, b))
                else:
                    c.append((b, a))
            c.extend(bitonic_merge_sub(hh, off))
            c.extend(bitonic_merge_sub(hh, off + hh))
            return c
        comps.extend(bitonic_merge_sub(n, offset))
        return comps
    
    try:
        net3 = bitonic_sort(N)
        candidates.append(net3)
    except Exception:
        pass
    
    # Verify all candidates and pick the best
    best_net = None
    best_count = float('inf')
    
    for net in candidates:
        # Deduplicate comparators
        seen = set()
        deduped = []
        for c in net:
            key = (min(c), max(c))
            if key not in seen:
                seen.add(key)
                deduped.append(key)
        
        if not _verify(deduped, N):
            continue
        
        # Try to minimize
        minimized = _dedup_and_minimize(deduped, N)
        
        if len(minimized) < best_count:
            best_count = len(minimized)
            best_net = minimized
    
    if best_net is None:
        # Fallback: odd-even transposition
        best_net = [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]
    
    return best_net
# EVOLVE-BLOCK-END
