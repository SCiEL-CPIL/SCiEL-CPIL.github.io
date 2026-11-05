"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 50.0
    
    # Fast input generation via periodic tiling
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for col in range(N):
        period = 1 << (col + 1)
        half = 1 << col
        block = np.zeros(period, dtype=np.uint8)
        block[half:] = 1
        repeats = (1 << N) // period
        inputs[:, col] = np.tile(block, repeats)
    
    state_buffer = np.empty_like(inputs)
    
    def verify(net):
        """Verify network sorts all inputs. Returns True if sorted."""
        np.copyto(state_buffer, inputs)
        for i, j in net:
            mi = state_buffer[:, i] & state_buffer[:, j]
            ma = state_buffer[:, i] | state_buffer[:, j]
            state_buffer[:, i] = mi
            state_buffer[:, j] = ma
        return np.all(state_buffer[:, :-1] <= state_buffer[:, 1:])
    
    def verify_partial(net, critical_indices):
        """Verify network only on critical (previously unsorted) inputs."""
        if len(critical_indices) == 0:
            return True, set()
        sub_inputs = inputs[critical_indices]
        state = sub_inputs.copy()
        for i, j in net:
            mi = state[:, i] & state[:, j]
            ma = state[:, i] | state[:, j]
            state[:, i] = mi
            state[:, j] = ma
        unsorted = np.any(state[:, :-1] > state[:, 1:], axis=1)
        if not np.any(unsorted):
            return True, set()
        new_critical = set(critical_indices[unsorted].tolist())
        return False, new_critical
    
    def _odd_even_phases():
        """Return odd-even transposition as list of phases (each phase is a set of (i,j) pairs)."""
        phases = []
        for phase in range(N):
            comps = [(i, i + 1) for i in range(phase % 2, N - 1, 2)]
            phases.append(comps)
        return phases
    
    def phases_to_flat(phases):
        """Convert phases to flat list of comparators."""
        flat = []
        for phase in phases:
            flat.extend(phase)
        return flat
    
    def flat_to_phases(flat):
        """Convert flat list to phases by grouping non-conflicting comparators."""
        if not flat:
            return []
        phases = []
        used = [False] * len(flat)
        remaining = list(flat)
        while remaining:
            current_phase = []
            wires_used = set()
            new_remaining = []
            for comp in remaining:
                i, j = comp
                if i not in wires_used and j not in wires_used:
                    current_phase.append(comp)
                    wires_used.add(i)
                    wires_used.add(j)
                else:
                    new_remaining.append(comp)
            if current_phase:
                phases.append(current_phase)
            remaining = new_remaining
        return phases
    
    def optimize_phase_comparators(phase_comps, rng):
        """Try to optimize comparators within a phase by swapping to better positions."""
        if not phase_comps:
            return phase_comps
        new_comps = list(phase_comps)
        # Try to shift comparators to more useful positions
        for idx in range(len(new_comps)):
            i, j = new_comps[idx]
            # Try nearby positions
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < nj < N:
                        # Check no conflict within phase
                        conflict = False
                        for k, (ci, cj) in enumerate(new_comps):
                            if k == idx:
                                continue
                            if ni in (ci, cj) or nj in (ci, cj):
                                conflict = True
                                break
                        if not conflict:
                            new_comps[idx] = (ni, nj)
                            break
        return new_comps
    
    def remove_comparator_from_phases(phases, comp_idx, rng):
        """Remove a comparator and try to redistribute."""
        new_phases = [list(p) for p in phases]
        # Find which phase contains this comparator
        total_idx = 0
        for pi, phase in enumerate(new_phases):
            if total_idx + len(phase) > comp_idx:
                local_idx = comp_idx - total_idx
                new_phases[pi].pop(local_idx)
                break
            total_idx += len(phase)
        # Remove empty phases
        new_phases = [p for p in new_phases if p]
        return new_phases
    
    # Start with odd-even transposition
    initial_phases = _odd_even_phases()
    best_flat = phases_to_flat(initial_phases)
    best_count = len(best_flat)
    best_phases = [list(p) for p in initial_phases]
    
    if not verify(best_flat):
        return best_flat
    
    # Get initial critical set (should be empty if verified)
    _, critical_set = verify_partial(best_flat, set(range(1 << N)))
    
    current_phases = [list(p) for p in best_phases]
    current_flat = phases_to_flat(current_phases)
    current_count = len(current_flat)
    
    rng = np.random.RandomState(42)
    last_improvement = time.time()
    
    while time.time() < deadline:
        improved = False
        
        # Strategy 1: Try removing comparators (greedy)
        if current_count > 0:
            # Try random removals
            for _ in range(min(20, current_count)):
                idx = rng.randint(current_count)
                candidate_phases = [list(p) for p in current_phases]
                candidate_flat = phases_to_flat(candidate_phases)
                
                # Remove comparator at idx
                total = 0
                for pi in range(len(candidate_phases)):
                    if total + len(candidate_phases[pi]) > idx:
                        local_idx = idx - total
                        candidate_phases[pi].pop(local_idx)
                        break
                    total += len(candidate_phases[pi])
                
                candidate_phases = [p for p in candidate_phases if p]
                candidate_flat = phases_to_flat(candidate_phases)
                
                if verify(candidate_flat):
                    current_phases = candidate_phases
                    current_flat = candidate_flat
                    current_count = len(current_flat)
                    improved = True
                    last_improvement = time.time()
                    if current_count < best_count:
                        best_phases = [list(p) for p in current_phases]
                        best_flat = list(current_flat)
                        best_count = current_count
                    break
        
        # Strategy 2: If stuck, try phase-level mutations
        if not improved and time.time() - last_improvement > 2.0:
            # Mutate a random phase
            if current_phases:
                pi = rng.randint(len(current_phases))
                mutated_phases = [list(p) for p in current_phases]
                mutated_phases[pi] = optimize_phase_comparators(mutated_phases[pi], rng)
                candidate_flat = phases_to_flat(mutated_phases)
                
                if verify(candidate_flat):
                    current_phases = mutated_phases
                    current_flat = candidate_flat
                    last_improvement = time.time()
        
        # Strategy 3: If very stuck, try adding a comparator to escape
        if not improved and time.time() - last_improvement > 5.0 and current_count < 100:
            # Try adding a comparator in a new phase
            new_i = rng.randint(N)
            new_j = rng.randint(N)
            if new_i == new_j:
                continue
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            
            candidate_phases = [list(p) for p in current_phases]
            candidate_phases.append([(new_i, new_j)])
            candidate_flat = phases_to_flat(candidate_phases)
            
            if verify(candidate_flat):
                # Accept with some probability (annealing-like)
                if rng.random() < 0.3:
                    current_phases = candidate_phases
                    current_flat = candidate_flat
                    current_count = len(current_flat)
        
        # Safety: if we've been stuck too long, reset to best
        if time.time() - last_improvement > 10.0:
            current_phases = [list(p) for p in best_phases]
            current_flat = list(best_flat)
            current_count = best_count
            last_improvement = time.time()
    
    return best_flat

# EVOLVE-BLOCK-END
