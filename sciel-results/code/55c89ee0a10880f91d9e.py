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
    # More efficient: for each wire i, the mask has bit k set iff (k >> i) & 1
    # This is a periodic pattern: 0,1,0,1,... with period 2^(i+1)
    for i in range(N):
        period = 1 << (i + 1)
        half = 1 << i
        # Build the mask by setting bits in pattern
        m = 0
        for block_start in range(0, NUM_INPUTS, period):
            for offset in range(half):
                m |= (1 << (block_start + half + offset))
        masks[i] = m
    return masks

# Precompute once at module load to save time in construct()
_WIRE_MASKS = _build_wire_bitmasks()


def _is_sorted(state):
    """Check if the network state is sorted for all inputs.
    Sorted means wire 0 <= wire 1 <= ... <= wire N-1 for every input.
    In bitmask terms: state[i] is a subset of state[i+1] for all i.
    """
    for i in range(N - 1):
        if (state[i] & ~state[i+1]) != 0:
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
    1. Start with odd-even transposition seed.
    2. Use bitmask simulation with incremental state tracking for rapid pruning.
    3. Multiple pruning passes with forward and backward strategies.
    
    Returns: list of (i, j) pairs.
    """
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
        return network
    
    # Efficient pruning using prefix/suffix state caching
    def _prune_network(comps):
        """Prune redundant comparators using incremental state tracking."""
        comps = list(comps)
        n_comps = len(comps)
        
        # Precompute prefix states: prefix_states[k] = state after applying comps[0..k-1]
        # prefix_states[0] = initial state
        prefix_states = [None] * (n_comps + 1)
        prefix_states[0] = list(_WIRE_MASKS)
        for k in range(n_comps):
            st = list(prefix_states[k])
            i, j = comps[k]
            _apply_comparator(st, i, j)
            prefix_states[k + 1] = st
        
        # Precompute suffix states: suffix_states[k] = state after applying comps[k..n-1] starting from identity
        # Actually, we need: if we remove comp k, we apply comps[0..k-1] then comps[k+1..n-1]
        # So suffix_states[k] should be the transformation of comps[k..n-1]
        # But that's not straightforward with bitmasks.
        
        # Alternative: for each k, compute state after removing comp k:
        # state_after_remove_k = apply comps[k+1..n-1] to prefix_states[k]
        # We can precompute "suffix transforms" but they're not simple to compose.
        
        # Better approach: just re-simulate from prefix_states[k] through remaining comparators.
        # To speed up, we can use the fact that if prefix_states[k] is already sorted,
        # then the remaining comparators don't matter.
        
        changed = True
        while changed:
            changed = False
            n_comps = len(comps)
            # Rebuild prefix states
            prefix_states = [None] * (n_comps + 1)
            prefix_states[0] = list(_WIRE_MASKS)
            for k in range(n_comps):
                st = list(prefix_states[k])
                i, j = comps[k]
                _apply_comparator(st, i, j)
                prefix_states[k + 1] = st
            
            for k in range(n_comps):
                # Check if removing comps[k] still gives sorted output
                # Start from prefix_states[k], apply comps[k+1..]
                st = list(prefix_states[k])
                for m in range(k + 1, n_comps):
                    i, j = comps[m]
                    _apply_comparator(st, i, j)
                if _is_sorted(st):
                    comps = comps[:k] + comps[k+1:]
                    changed = True
                    break  # Restart
        
        return comps
    
    network = _prune_network(network)
    
    # Second pass: try backward pruning (remove from end first)
    def _prune_network_backward(comps):
        comps = list(comps)
        changed = True
        while changed:
            changed = False
            n_comps = len(comps)
            for k in range(n_comps - 1, -1, -1):
                reduced = comps[:k] + comps[k+1:]
                st = list(_WIRE_MASKS)
                for i, j in reduced:
                    _apply_comparator(st, i, j)
                if _is_sorted(st):
                    comps = reduced
                    changed = True
                    break
        return comps
    
    network = _prune_network_backward(network)
    
    # Third pass: forward again
    network = _prune_network(network)
    
    return network
