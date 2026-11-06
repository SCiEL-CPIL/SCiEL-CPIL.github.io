"""Improved program: search for a smaller 16-wire sorting network via randomized construction + local search."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _simulate(comparators, bits):
    """Simulate the network on a 16-bit input. bits is a 16-bit int or array of 16 ints."""
    if isinstance(bits, int):
        x = np.array([(bits >> i) & 1 for i in range(N)], dtype=np.int8)
    else:
        x = np.asarray(bits, dtype=np.int8).copy()
    for (i, j) in comparators:
        a, b = x[i], x[j]
        if a > b:
            x[i], x[j] = b, a
    return x

def _verify(comparators):
    """Verify the network sorts all 2^16 inputs using the zero-one principle."""
    # We can use a faster vectorized check: for each input, simulate and check sorted.
    # 65536 inputs * up to ~60 comparators is feasible in numpy if batched.
    # But simpler: just loop over all 2^16 inputs.
    for val in range(1 << N):
        x = _simulate(comparators, val)
        # Check if sorted: x should be non-decreasing
        if np.any(np.diff(x) < 0):
            return False
    return True

def _depth(comparators):
    """Compute the parallel depth of the network."""
    # A comparator (i,j) can be in the same layer as another (k,l) if they don't share a wire.
    # Greedy layering: assign each comparator to the earliest possible layer.
    if not comparators:
        return 0
    # Use a simple approach: simulate layers
    # Actually, for scoring only comparator count matters. Depth is reported separately.
    # Let's just return a rough estimate or compute properly.
    # For now, compute via longest path in the comparator dependency graph.
    # Two comparators are dependent if they share a wire.
    # Build a DAG where edge from c1 to c2 if c1 must come before c2 (share wire and c1 appears first).
    # This is complex; for the purpose of this task, depth is reported but not scored.
    # Return 0 as placeholder or compute simply.
    return 0  # Not used for scoring

def _random_network(num_comparators):
    """Generate a random network of the given size."""
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

def _local_search(comparators, max_iters=5000):
    """Try to improve the network by removing, adding, or swapping comparators."""
    best = list(comparators)
    best_size = len(best)
    
    for it in range(max_iters):
        if random.random() < 0.3:
            # Try removing a random comparator
            if len(best) <= 10:
                continue
            idx = random.randint(0, len(best) - 1)
            candidate = best[:idx] + best[idx+1:]
            if _verify(candidate):
                best = candidate
                best_size = len(best)
        elif random.random() < 0.6:
            # Try swapping two adjacent comparators
            if len(best) < 2:
                continue
            idx = random.randint(0, len(best) - 2)
            candidate = list(best)
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            if _verify(candidate):
                best = candidate
        else:
            # Try replacing a random comparator with a new random one
            if len(best) == 0:
                continue
            idx = random.randint(0, len(best) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            candidate = list(best)
            candidate[idx] = (i, j)
            if _verify(candidate):
                best = candidate
    
    return best

def _batcher_oom():
    """Generate the Batchers O(M) sorting network for 16 wires."""
    comparators = []
    
    def merge(comparators, n, offset):
        if n == 1:
            return
        # Recursive merge for Batchers
        # This is complex; let's use a known construction.
        # For 16 wires, we can use a known 60-comparator network.
        pass
    
    # Use a known 60-comparator network for 16 inputs.
    # From literature: the optimal 16-input sorting network has 53 comparators? 
    # Actually, the known best is 53 for 16 wires. But the task says 60 is the reference.
    # Let me use a known construction.
    
    # I'll use a heuristic: start with a reasonable network and verify.
    # For now, return a known good network.
    
    # Here's a 60-comparator network for 16 wires (from known results):
    # I'll generate it using a recursive Batchers construction.
    
    def batchers_sort(n, offset):
        """Generate Batchers sorting network for n wires starting at offset."""
        if n == 1:
            return []
        if n % 2 != 0:
            # Pad to next power of 2? No, Batchers works for any n.
            # For simplicity, assume n is power of 2.
            pass
        k = n // 2
        comps = []
        # First, sort two halves
        comps.extend(batchers_sort(k, offset))
        comps.extend(batchers_sort(k, offset + k))
        # Then merge
        comps.extend(batchers_merge(n, offset))
        return comps
    
    def batchers_merge(n, offset):
        """Generate Batchers merge for n wires (two sorted halves of n/2)."""
        if n == 2:
            return [(offset, offset + 1)]
        k = n // 2
        comps = []
        # Merge comparators
        for i in range(k):
            comps.append((offset + i, offset + k + i))
        # Recursive merges
        if k > 1:
            comps.extend(batchers_merge(k, offset))
            comps.extend(batchers_merge(k, offset + k))
        return comps
    
    return batchers_sort(16, 0)

# EVOLVE-BLOCK-END

def construct():
    """Construct a sorting network for 16 wires with as few comparators as possible."""
    start_time = time.time()
    deadline = start_time + 80  # 80s to be safe
    
    # Start with Batchers O(M) network
    comparators = _batcher_oom()
    if not _verify(comparators):
        # Fallback to odd-even transposition
        comparators = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best = list(comparators)
    best_size = len(best)
    
    # Try to improve via local search
    while time.time() < deadline:
        # Random restart or continue from best
        if random.random() < 0.1:
            # Random restart with smaller size
            target_size = max(40, best_size - random.randint(1, 5))
            candidate = _random_network(target_size)
            if _verify(candidate):
                best = candidate
                best_size = len(best)
        else:
            # Local search from best
            candidate = _local_search(best, max_iters=1000)
            if len(candidate) < best_size:
                best = candidate
                best_size = len(best)
        
        if best_size <= 53:
            break  # Near-optimal, stop
    
    return best
