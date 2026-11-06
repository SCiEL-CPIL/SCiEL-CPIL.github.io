"""Size-efficient sorting network on 16 wires via local search with zero-one verification."""
import time
import random

N = 16

def _make_verify_fn():
    """Return a function that checks if the network sorts all 65536 inputs."""
    
    def _is_sorted(x):
        k = bin(x).count('1')
        if k == 0:
            return x == 0
        if k == 16:
            return x == 0xFFFF
        return (x >> (16 - k)) == (1 << k) - 1 and (x & ((1 << (16 - k)) - 1)) == 0
    
    def _apply(network, x):
        for (i, j) in network:
            bi = (x >> i) & 1
            bj = (x >> j) & 1
            if bi and not bj:
                x = x & ~(1 << i) | (1 << j)
        return x
    
    def _verify(network):
        for x in range(65536):
            result = _apply(network, x)
            if not _is_sorted(result):
                return False
        return True
    
    return _verify

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    verify = _make_verify_fn()
    
    # Start with odd-even transposition (120 comparators)
    best_network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    best_count = len(best_network)
    
    ok = verify(best_network)
    if not ok:
        return best_network
    
    current_network = list(best_network)
    current_count = best_count
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Try to remove a random comparator
        if current_count > 50:
            idx = rng.randint(0, current_count - 1)
            saved = current_network.pop(idx)
            ok = verify(current_network)
            if ok:
                current_count -= 1
                if current_count < best_count:
                    best_count = current_count
                    best_network = list(current_network)
            else:
                current_network.insert(idx, saved)
        
        # Try to swap two comparators
        if current_count > 1:
            idx = rng.randint(0, current_count - 1)
            j = rng.randint(0, current_count - 1)
            if idx != j:
                current_network[idx], current_network[j] = current_network[j], current_network[idx]
                ok = verify(current_network)
                if not ok:
                    current_network[idx], current_network[j] = current_network[j], current_network[idx]
        
        # Occasionally try removing multiple comparators in sequence
        if current_count > 60 and rng.random() < 0.1:
            for _ in range(3):
                if current_count <= 50:
                    break
                idx = rng.randint(0, current_count - 1)
                saved = current_network.pop(idx)
                ok = verify(current_network)
                if ok:
                    current_count -= 1
                    if current_count < best_count:
                        best_count = current_count
                        best_network = list(current_network)
                else:
                    current_network.insert(idx, saved)
                    break
    
    return best_network
