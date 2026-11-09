"""Size-efficient sorting network on 17 wires using bitmask-based pruning."""
import numpy as np

N = 17
NUM_INPUTS = 1 << N  # 131,072

# Precompute the bitmask for each input wire.
# wire_bitmask[i] is an integer of NUM_INPUTS bits where bit k is 1 iff wire i is 1 in input k.
# We represent the state of each wire as a Python int (bitmask over all inputs).

def _build_wire_bitmasks():
    """Build the initial bitmask for each wire: bit k of wire i is 1 iff input k has bit i set."""
    masks = [0] * N
    for k in range(NUM_INPUTS):
        for i in range(N):
            if (k >> i) & 1:
                masks[i] |= (1 << k)
    return masks

# Precompute once at module load to save time in construct()
_WIRE_MASKS = _build_wire_bitmasks()


def _is_sorted(state):
    """Check if the network state is sorted for all inputs.
    state[i] is the bitmask of wire i.
    Sorted means: for all i < j, state[i] & state[j] == 0 (no input has 1 in wire i and 0 in wire j when i<j... 
    actually: for sorted order, wire 0 should be the min, so state[0] should be subset of all others? 
    No: sorted means wire 0 <= wire 1 <= ... <= wire N-1 for every input.
    In bitmask terms: state[i] & ~state[i+1] == 0 for all i, meaning if wire i is 1 then wire i+1 must be 1.
    Equivalently: state[i] | state[i+1] == state[i+1], i.e., state[i] is a subset of state[i+1].
    """
    for i in range(N - 1):
        if (state[i] | state[i+1]) != state[i+1]:
            return False
    return True


def _apply_comparator(state, i, j):
    """Apply compare-exchange on wires i and j (i < j).
    min goes to i, max goes to j.
    min(a,b) = a & b, max(a,b) = a | b
    """
    new_i = state[i] & state[j]
    new_j = state[i] | state[j]
    state[i] = new_i
    state[j] = new_j
    return state


def construct():
    """Construct a sorting network for N=17 wires using bitmask-guided pruning.
    
    Strategy:
    1. Start with a reasonable seed network (e.g., odd-even transposition or a known good network).
    2. Use bitmask simulation to verify correctness.
    3. Prune redundant comparators by checking if removing each comparator still yields a valid sorting network.
    4. Additionally, try to find shorter networks by starting from a smaller seed or using greedy construction.
    
    Returns: list of (i, j) pairs.
    """
    # Step 1: Generate a seed network.
    # Use odd-even transposition as a baseline, but we can try to start from a better seed.
    # For N=17, odd-even transposition uses N*N = 289 comparators? No, it's N phases of ~N/2 comparators.
    # Actually for N wires, odd-even transposition uses N * ceil(N/2) or so. Let's just generate it.
    
    def _odd_even_transposition(n):
        comps = []
        for phase in range(n):
            if phase % 2 == 0:
                for i in range(0, n - 1, 2):
                    comps.append((i, i + 1))
            else:
                for i in range(1, n - 1, 2):
                    comps.append((i, i + 1))
        return comps
    
    # Start with odd-even transposition
    network = _odd_even_transposition(N)
    
    # Verify it's correct
    state = list(_WIRE_MASKS)
    for i, j in network:
        _apply_comparator(state, i, j)
    if not _is_sorted(state):
        # Should not happen for odd-even transposition, but just in case
        return network
    
    # Step 2: Prune redundant comparators
    # Iterate through the network and try to remove each comparator if the network remains valid.
    # We do multiple passes since removing one comparator may make others redundant.
    
    def _validate_network(comps):
        st = list(_WIRE_MASKS)
        for i, j in comps:
            _apply_comparator(st, i, j)
        return _is_sorted(st)
    
    # Greedy forward pruning: try to remove each comparator in order
    changed = True
    while changed:
        changed = False
        for idx in range(len(network)):
            # Try removing comparator at idx
            reduced = network[:idx] + network[idx+1:]
            if _validate_network(reduced):
                network = reduced
                changed = True
                break  # Restart the scan since indices changed
    
    # Step 3: Try to find a better network by starting from a different seed or using a more aggressive search.
    # For now, return the pruned network.
    
    return network
