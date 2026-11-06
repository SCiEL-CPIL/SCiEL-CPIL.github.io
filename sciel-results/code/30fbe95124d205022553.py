"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Build a sorting network on N=18 wires using Python big-integer bit-parallel verification.
    Uses error-guided repair, move operations, and swap-based local search.
    """
    start_time = time.time()
    TOTAL_INPUTS = 1 << N  # 262144
    DEADLINE = start_time + 55.0  # Safety margin before 60s limit

    # --- Initialize Bit-Parallel Wires using Python big integers ---
    # wire[i] is a Python int where bit k is 1 if input k has 1 on wire i.
    # Input k is the integer k. Wire i corresponds to bit i of k.
    
    def make_mask(i):
        """Create mask for wire i: bit k is set if (k >> i) & 1 == 1."""
        L = 1 << (i + 1)
        ones_val = (1 << (1 << i)) - 1
        block = ones_val << (1 << i)
        G = ((1 << TOTAL_INPUTS) - 1) // ((1 << L) - 1)
        mask = block * G
        return mask

    # Pre-compute masks
    masks = [make_mask(i) for i in range(N)]

    def verify_network(network):
        """Verify network sorts all inputs. Returns True if correct."""
        wires = masks[:]
        for i, j in network:
            new_wi = wires[i] & wires[j]
            new_wj = wires[i] | wires[j]
            wires[i] = new_wi
            wires[j] = new_wj
        for i in range(N - 1):
            if (wires[i] & ~wires[i + 1]) != 0:
                return False
        return True

    def find_violation(network):
        """Find the first adjacent inversion pair, or None if sorted."""
        wires = masks[:]
        for i, j in network:
            new_wi = wires[i] & wires[j]
            new_wj = wires[i] | wires[j]
            wires[i] = new_wi
            wires[j] = new_wj
        for i in range(N - 1):
            if (wires[i] & ~wires[i + 1]) != 0:
                return (i, i + 1)
        return None

    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    # --- Start with odd-even transposition network ---
    network = odd_even_transposition()
    assert verify_network(network), "Initial network should be correct"

    best_network = network[:]
    best_size = len(network)

    # --- Phase 1: Aggressive Greedy Removal ---
    removed = True
    while removed and (time.time() < DEADLINE - 15):
        removed = False
        for idx in range(len(network)):
            if time.time() > DEADLINE - 12:
                break
            candidate = network[:]
            candidate.pop(idx)
            if verify_network(candidate):
                network = candidate
                if len(network) < best_size:
                    best_network = network[:]
                    best_size = len(network)
                removed = True
                break  # Restart since indices changed

    # --- Phase 2: Error-guided local search with move operations ---
    current = best_network[:]
    
    def try_move(current_net, src_idx, dst_idx):
        """Try moving comparator at src_idx to position dst_idx."""
        if src_idx == dst_idx:
            return None
        candidate = current_net[:]
        comp = candidate.pop(src_idx)
        # Adjust dst_idx if needed
        if src_idx < dst_idx:
            dst_idx -= 1
        candidate.insert(dst_idx, comp)
        return candidate

    for iteration in range(60000):
        if time.time() > DEADLINE - 8:
            break
        if len(current) <= 72:
            break
            
        op_type = random.random()
        
        if op_type < 0.4:
            # Random removal attempt
            idx = random.randint(0, len(current) - 1)
            candidate = current[:]
            candidate.pop(idx)
            
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
                # Try inserting comparators involving the violated wires
                repair_candidates = []
                for w in range(N):
                    if w != vi and w != vj:
                        if w < vi:
                            repair_candidates.append((w, vi))
                        else:
                            repair_candidates.append((vi, w))
                    if w != vj:
                        if w < vj:
                            repair_candidates.append((w, vj))
                        else:
                            repair_candidates.append((vj, w))
                repair_candidates.append((vi, vj))
                
                random.shuffle(repair_candidates)
                
                for (i, j) in repair_candidates[:20]:
                    if time.time() > DEADLINE - 8:
                        break
                    # Try inserting at positions near the violation
                    # Prefer positions after the last comparator involving vi or vj
                    last_pos = -1
                    for pos, (ci, cj) in enumerate(candidate):
                        if ci == vi or cj == vi or ci == vj or cj == vj:
                            last_pos = pos
                    insert_positions = set()
                    insert_positions.add(last_pos + 1 if last_pos >= 0 else 0)
                    insert_positions.add(len(candidate))
                    insert_positions.add(random.randint(0, len(candidate)))
                    if last_pos >= 0:
                        insert_positions.add(max(0, last_pos - 1))
                    
                    for pos in list(insert_positions)[:4]:
                        pos = min(pos, len(candidate))
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
                    continue
            else:
                continue
                
        elif op_type < 0.7:
            # Move operation
            if len(current) < 2:
                continue
            src_idx = random.randint(0, len(current) - 1)
            dst_idx = random.randint(0, len(current) - 1)
            candidate = try_move(current, src_idx, dst_idx)
            if candidate is None:
                continue
            if verify_network(candidate):
                current = candidate
                if len(current) < best_size:
                    best_network = current[:]
                    best_size = len(current)
                    
        else:
            # Swap operation
            if len(current) < 2:
                continue
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

    # --- Phase 3: Final Greedy Removal with multiple passes ---
    final_net = best_network[:]
    for _ in range(3):
        removed = True
        while removed and (time.time() < DEADLINE - 3):
            removed = False
            for idx in range(len(final_net)):
                if time.time() > DEADLINE - 2:
                    break
                candidate = final_net[:]
                candidate.pop(idx)
                if verify_network(candidate):
                    final_net = candidate
                    removed = True
                    break

    if len(final_net) < best_size:
        best_network = final_net[:]
        best_size = len(final_net)

    # Final verification
    if not verify_network(best_network):
        best_network = odd_even_transposition()

    return best_network
# EVOLVE-BLOCK-END
