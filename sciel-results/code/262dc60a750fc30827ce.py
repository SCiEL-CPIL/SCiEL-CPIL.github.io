"""HP lattice protein folding: Contact-graph guided local search with annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
# Safety margin: Evaluator kills at 210s. We target finishing by 205s.
_GLOBAL_DEADLINE = _GLOBAL_START + 205.0

_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            for dx, dy in _DELTAS:
                q = (x + dx, y + dy)
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _backtrack_fold(sequence, time_limit, rng):
    """
    Fold sequence using contact-graph guided initialization and local refinement.
    """
    n = len(sequence)
    if n < 2:
        return [(0, 0)] * n

    is_h = [c == 'H' for c in sequence]
    h_indices = [i for i in range(n) if is_h[i]]
    
    # Pre-compute potential contact pairs (i, j) where i < j, both H, and |i-j| > 1
    # We don't know geometry yet, but we know which pairs *could* form contacts.
    # This helps in biasing the search towards packing H's together.
    
    best_chain = None
    best_energy = 0

    # 1. Greedy Initialization with Contact-Aware Bias
    def _greedy_grow(seq, rng, hh_weight=1.0, free_weight=0.1, noise=0.3):
        nn = len(seq)
        if nn < 2:
            return [(0, 0)] * nn
        start = (0, 0)
        chain = [start]
        occupied = {start: 0}
        
        # Track which H's are "exposed" or have high potential
        # Simple heuristic: prefer placing H near existing H's
        
        for i in range(1, nn):
            options = []
            x, y = chain[-1]
            for site in _DELTAS:
                cand = (x + site[0], y + site[1])
                if cand in occupied:
                    continue
                
                gain = 0
                if seq[i] == 'H':
                    cx, cy = cand
                    for dx, dy in _DELTAS:
                        q = (cx + dx, cy + dy)
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and seq[j] == 'H':
                            gain += 1
                            
                # Also look ahead: does this position allow future H's to contact?
                # Rough estimate: count free neighbors that are not blocked by chain
                free = 0
                cx, cy = cand
                for dx, dy in _DELTAS:
                    if (cx + dx, cy + dy) not in occupied:
                        free += 1
                        
                # Bias: H-H contacts are worth more, but don't trap the chain
                score = hh_weight * gain + free_weight * free + noise * rng.random()
                options.append((score, cand))
                
            if not options:
                return None
            
            # Sort by score descending, pick top with some randomness
            options.sort(key=lambda x: x[0], reverse=True)
            # Pick from top 2-3 to avoid deterministic traps
            if len(options) > 1 and rng.random() < 0.3:
                choice_idx = min(len(options) - 1, rng.randint(0, 2))
                best_score, best_site = options[choice_idx]
            else:
                best_score, best_site = options[0]
                
            occupied[best_site] = i
            chain.append(best_site)

        return chain

    # Generate diverse initial solutions
    initial_chains = []
    configs = [
        (2.0, 0.0, 0.5),  # High H bias
        (1.0, 0.5, 0.5),  # Balanced
        (0.5, 1.0, 0.5),  # Free space bias (unfolded)
        (1.5, 0.2, 0.3),  # Moderate H
        (1.0, 0.3, 1.0),  # Noisy
        (3.0, 0.0, 0.2),  # Very high H
    ]
    
    growth_deadline = time.time() + min(2.0, time_limit * 0.1)
    for hh_w, free_w, noise in configs:
        if time.time() > growth_deadline:
            break
        for attempt in range(2):
            if time.time() > growth_deadline:
                break
            chain = _greedy_grow(sequence, rng, hh_w, free_w, noise)
            if chain is not None:
                initial_chains.append(chain)

    if not initial_chains:
        initial_chains.append([(i, 0) for i in range(n)])

    # Evaluate and sort initial chains
    scored = []
    for chain in initial_chains:
        e = _energy(sequence, chain)
        scored.append((e, chain))
    scored.sort(key=lambda x: x[0])
    candidates = [s[1] for s in scored[:min(4, len(scored))]]

    # 2. Local Refinement with Contact-Aware Moves
    def _local_refine(chain, time_limit, rng):
        """
        Iterated Local Search (ILS) with Simulated Annealing.
        Moves: Reversals and Pivots.
        Bias: Moves that preserve or create H-H contacts are favored.
        """
        nn = len(chain)
        current_energy = _energy(sequence, chain)
        best_energy = current_energy
        best_chain = chain[:]
        
        T = 1.5
        cooling = 0.9999
        min_t = 0.01
        
        deadline = time.time() + time_limit
        no_improve = 0
        max_no_improve = 2000
        
        # Precompute H indices for bias
        h_indices = [i for i in range(nn) if sequence[i] == 'H']
        
        def _try_reversal(state_chain, i, j):
            if i >= j:
                return None
            # Check boundary constraints for self-avoidance
            if i > 0:
                px, py = state_chain[i - 1]
                ex, ey = state_chain[j]
                if abs(px - ex) + abs(py - ey) != 1:
                    return None
            if j < nn - 1:
                sx, sy = state_chain[i]
                nx, ny = state_chain[j + 1]
                if abs(sx - nx) + abs(sy - ny) != 1:
                    return None
            
            # Collision check
            segment = state_chain[i:j+1]
            # Optimization: Only check collision with non-adjacent parts
            # For simplicity and safety, use set intersection
            new_chain = state_chain[:i] + segment[::-1] + state_chain[j+1:]
            if len(set(new_chain)) != nn:
                return None
            return new_chain

        def _try_pivot(state_chain, i, angle):
            if i <= 0 or i >= nn - 1:
                return None
            px, py = state_chain[i]
            tail = state_chain[i+1:]
            rotated = []
            for tx, ty in tail:
                rx, ry = tx - px, ty - py
                for _ in range(angle % 4):
                    rx, ry = -ry, rx
                rotated.append((px + rx, py + ry))
            new_chain = state_chain[:i+1] + rotated
            if len(set(new_chain)) != nn:
                return None
            return new_chain

        def _generate_move():
            # Bias towards H-rich regions for pivots/reversals
            # If we pivot near an H, we might create/break a contact.
            move_type = rng.random()
            if move_type < 0.6:
                # Reversal
                # Bias: Start/End reversal near H's
                if h_indices and rng.random() < 0.5:
                    hi = rng.choice(h_indices)
                    span = rng.randint(2, min(15, nn - 1))
                    i = max(0, hi - span // 2)
                    j = min(nn - 1, i + span)
                    if j <= i:
                        i = rng.randint(0, nn - 2)
                        j = rng.randint(i + 1, nn - 1)
                else:
                    i = rng.randint(0, nn - 2)
                    j = rng.randint(i + 1, nn - 1)
                return ('rev', i, j)
            else:
                # Pivot
                if h_indices and rng.random() < 0.5:
                    hi = rng.choice(h_indices)
                    i = max(1, min(nn - 2, hi + rng.randint(-2, 2)))
                else:
                    i = rng.randint(1, nn - 2)
                angle = rng.choice([1, 2, 3])
                return ('pivot', i, angle)

        while time.time() < deadline:
            if _remaining_time() < 2.0:
                break

            move = _generate_move()
            if move[0] == 'rev':
                new_chain = _try_reversal(chain, move[1], move[2])
            else:
                new_chain = _try_pivot(chain, move[1], move[2])

            if new_chain is None:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
                continue

            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy

            accept = False
            if delta <= 0:
                accept = True
            elif T > min_t:
                exp_arg = -delta / T
                if exp_arg > -50:
                    accept = rng.random() < math.exp(exp_arg)

            if accept:
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            if no_improve > max_no_improve:
                # Restart from best with perturbation
                if _remaining_time() < 3.0:
                    break
                chain = best_chain[:]
                current_energy = best_energy
                # Perturb
                for _ in range(rng.randint(3, 8)):
                    m = _generate_move()
                    if m[0] == 'rev':
                        nc = _try_reversal(chain, m[1], m[2])
                    else:
                        nc = _try_pivot(chain, m[1], m[2])
                    if nc is not None:
                        chain = nc
                        current_energy = _energy(sequence, chain)
                        break
                no_improve = 0
                T = max(T * 1.5, 0.5)
                # If we restart too many times, give up
                max_no_improve = int(max_no_improve * 0.8)
                if max_no_improve < 100:
                    break

            T *= cooling
            if T < min_t:
                break

        return best_chain, best_energy

    # 3. Time Allocation
    total_time = _remaining_time() - 3.0
    if total_time < 1.0:
        return candidates[0] if candidates else [(i, 0) for i in range(n)]

    # Allocate time to candidates. Better initial solutions get more time.
    # We have at most 4 candidates.
    # Weights: 50%, 30%, 15%, 5%
    weights = [0.5, 0.3, 0.15, 0.05]
    n_cands = len(candidates)
    if n_cands < len(weights):
        weights = weights[:n_cands]
        # Normalize if fewer candidates
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]
    
    for idx, chain in enumerate(candidates):
        if _remaining_time() < 2.0:
            break
        alloc = total_time * weights[idx]
        alloc = min(alloc, _remaining_time() - 1.0)
        if alloc < 0.5:
            continue
        
        refined_chain, refined_energy = _local_refine(chain, alloc, rng)
        
        if best_chain is None or refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain

    if best_chain is None:
        best_chain = candidates[0] if candidates else [(i, 0) for i in range(n)]

    return best_chain


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    rng = random.Random(42)
    n = len(sequence)

    if n < 2:
        return [[0, 0]] * n

    # Reserve 5s for validation and overhead
    time_limit = max(0.5, _remaining_time() - 5.0)

    best_chain = _backtrack_fold(sequence, time_limit, rng)

    # Validate: ensure self-avoiding and unit steps
    if best_chain is None or len(best_chain) != n:
        best_chain = [(i, 0) for i in range(n)]

    seen = set()
    valid = True
    for i, (x, y) in enumerate(best_chain):
        if (x, y) in seen:
            valid = False
            break
        seen.add((x, y))
        if i > 0:
            px, py = best_chain[i-1]
            if abs(px - x) + abs(py - y) != 1:
                valid = False
                break

    if not valid:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
