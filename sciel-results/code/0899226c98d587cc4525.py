"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 18
TOTAL_INPUTS = 1 << N  # 262144
DEADLINE = 55.0  # seconds, safety margin before 60s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Build a sorting network on N=18 wires using Bit-Parallel verification.
    Uses Python big integers to represent the state of each wire across all 2^18 inputs.
    min(a,b) is bitwise AND, max(a,b) is bitwise OR.
    Sorted check: (wire[i] & ~wire[i+1]) == 0 for all i.
    """
    start_time = time.time()

    # --- Initialize Bit-Parallel Wires ---
    def make_mask(i):
        L = 1 << (i + 1)
        num_blocks = TOTAL_INPUTS // L
        ones_val = (1 << (1 << i)) - 1
        block = ones_val << (1 << i)
        G = ((1 << TOTAL_INPUTS) - 1) // ((1 << L) - 1)
        mask = block * G
        return mask

    # Pre-compute masks
    masks = [make_mask(i) for i in range(N)]

    # Pre-compute the "all ones" mask for quick NOT
    ALL_ONES = (1 << TOTAL_INPUTS) - 1

    def is_sorted(wires):
        """Check if wires are sorted: wire[i] <= wire[i+1] for all i."""
        for i in range(N - 1):
            if (wires[i] & ~wires[i+1]) != 0:
                return False
        return True

    def verify_network(network):
        """Verify network sorts all inputs. Returns True if correct."""
        wires = masks[:]
        for i, j in network:
            a = wires[i]
            b = wires[j]
            wires[i] = a & b
            wires[j] = a | b
        return is_sorted(wires)

    def find_violation(network):
        """Find the first adjacent inversion pair, or None if sorted."""
        wires = masks[:]
        for i, j in network:
            a = wires[i]
            b = wires[j]
            wires[i] = a & b
            wires[j] = a | b
        for i in range(N - 1):
            if (wires[i] & ~wires[i+1]) != 0:
                return (i, i + 1)
        return None

    # --- Start with odd-even transposition network ---
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))

    assert verify_network(network), "Initial network should be correct"

    best_network = network[:]
    best_size = len(network)

    # --- Phase 1: Aggressive Greedy Removal ---
    # Try removing from the end first (often less critical), then scan
    def greedy_remove(net, time_limit):
        """Repeatedly remove any removable comparator. Returns improved network."""
        n = net[:]
        changed = True
        while changed and (time.time() - start_time) < time_limit:
            changed = False
            # Try from end to start (comparators added later may be less essential)
            for idx in range(len(n) - 1, -1, -1):
                if (time.time() - start_time) > time_limit:
                    break
                candidate = n[:]
                candidate.pop(idx)
                if verify_network(candidate):
                    n = candidate
                    changed = True
                    break
        return n

    best_network = greedy_remove(best_network, DEADLINE - 15)
    best_size = len(best_network)

    # --- Phase 2: Error-guided local search with broad repair ---
    current = best_network[:]
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    for iteration in range(30000):
        if (time.time() - start_time) > DEADLINE - 8:
            break
        if len(current) <= 70:
            break
            
        # Random removal
        idx = random.randint(0, len(current) - 1)
        candidate = current[:]
        candidate.pop(idx)
        
        # If it's still valid, accept
        if verify_network(candidate):
            current = candidate
            if len(current) < best_size:
                best_network = current[:]
                best_size = len(current)
            continue
            
        # If broken, find the violation and try to repair
        violation = find_violation(candidate)
        repaired = False
        if violation:
            vi, vj = violation
            # Build repair candidates: all pairs involving vi or vj, plus (vi, vj)
            repair_candidates = set()
            for w in range(N):
                if w != vi:
                    a, b = (w, vi) if w < vi else (vi, w)
                    repair_candidates.add((a, b))
                if w != vj:
                    a, b = (w, vj) if w < vj else (vj, w)
                    repair_candidates.add((a, b))
            repair_candidates.add((vi, vj))
            repair_list = list(repair_candidates)
            random.shuffle(repair_list)
            
            for (i, j) in repair_list[:30]:
                if (time.time() - start_time) > DEADLINE - 5:
                    break
                # Try inserting at multiple positions
                positions = [idx, max(0, idx-2), min(len(candidate), idx+2), 0, len(candidate),
                             max(0, idx-1), min(len(candidate), idx+1)]
                for pos in positions:
                    pos = max(0, min(len(candidate), pos))
                    temp = candidate[:]
                    temp.insert(pos, (i, j))
                    if verify_network(temp):
                        current = temp
                        repaired = True
                        if len(current) < best_size:
                            best_network = current[:]
                            best_size = len(current)
                        break
                if repaired:
                    break
        
        if not repaired:
            # Fallback: try a few random insertions
            for _ in range(8):
                if (time.time() - start_time) > DEADLINE - 5:
                    break
                (i, j) = random.choice(all_pairs)
                pos = random.randint(0, len(candidate))
                temp = candidate[:]
                temp.insert(pos, (i, j))
                if verify_network(temp):
                    current = temp
                    repaired = True
                    if len(current) < best_size:
                        best_network = current[:]
                        best_size = len(current)
                    break

    # --- Phase 3: Random swap-based local search ---
    current = best_network[:]
    for iteration in range(15000):
        if (time.time() - start_time) > DEADLINE - 5:
            break
        if len(current) <= 70:
            break
        if len(current) < 2:
            break
            
        i1 = random.randint(0, len(current) - 1)
        i2 = random.randint(0, len(current) - 1)
        if i1 == i2:
            continue
            
        candidate = current[:]
        candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
        
        if verify_network(candidate):
            current = candidate
            if len(current) < best_size:
                best_network = current[:]
                best_size = len(current)

    # --- Phase 4: Multi-restart local search from perturbed best ---
    # Try a few restarts with random perturbations to escape local optima
    for restart in range(3):
        if (time.time() - start_time) > DEADLINE - 6:
            break
        
        # Perturb: randomly swap 2-5 comparators
        perturbed = best_network[:]
        n_swaps = random.randint(2, 5)
        for _ in range(n_swaps):
            if len(perturbed) < 2:
                break
            a = random.randint(0, len(perturbed) - 1)
            b = random.randint(0, len(perturbed) - 1)
            if a != b:
                perturbed[a], perturbed[b] = perturbed[b], perturbed[a]
        
        # Run a short local search from perturbed state
        current = perturbed[:]
        for iteration in range(5000):
            if (time.time() - start_time) > DEADLINE - 4:
                break
            if len(current) <= 70:
                break
            if len(current) < 2:
                break
            
            # Mix of removal and swap moves
            if random.random() < 0.5:
                # Try removal
                idx = random.randint(0, len(current) - 1)
                candidate = current[:]
                candidate.pop(idx)
                if verify_network(candidate):
                    current = candidate
                    if len(current) < best_size:
                        best_network = current[:]
                        best_size = len(current)
            else:
                # Try swap
                i1 = random.randint(0, len(current) - 1)
                i2 = random.randint(0, len(current) - 1)
                if i1 == i2:
                    continue
                candidate = current[:]
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                if verify_network(candidate):
                    current = candidate
                    if len(current) < best_size:
                        best_network = current[:]
                        best_size = len(current)

    # --- Phase 5: Final Greedy Removal ---
    final_net = greedy_remove(best_network, DEADLINE - 1)

    if len(final_net) < best_size:
        best_network = final_net[:]
        best_size = len(final_net)

    return best_network
# EVOLVE-BLOCK-END
