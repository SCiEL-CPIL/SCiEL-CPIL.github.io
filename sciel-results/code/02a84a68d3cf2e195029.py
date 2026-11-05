"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
import random

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses bitset-based row construction combined with local search (row rebuilding)
    and simulated annealing to tighten the maximum entry.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    def time_up():
        return time.time() > deadline
    
    # Counting bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = 0
    while lb * (lb + 1) // 2 < total_diffs_needed:
        lb += 1
    
    # Bitset-based row builder: builds one row given forbidden diffs and max limit
    def build_row_bitset(k, forbidden_bits, max_val, rng, bias=0.0):
        """
        Build a row of k+1 elements starting at 0, strictly increasing,
        with all elements <= max_val, avoiding differences in forbidden_bits.
        Returns (row, diff_bits) or (None, 0).
        """
        row = [0]
        local_bits = 0
        
        for pos in range(k):
            last = row[-1]
            remaining = k - pos - 1
            max_x = max_val - remaining  # leave room for remaining elements
            
            if last + 1 > max_x:
                return None, 0
            
            best_x = None
            best_bits = 0
            best_score = float('inf')
            
            # Collect candidates
            candidates = []
            for x in range(last + 1, max_x + 1):
                new_bits = 0
                conflict = False
                for a in row:
                    d = x - a
                    if (forbidden_bits >> d) & 1:
                        conflict = True
                        break
                    new_bits |= (1 << d)
                if conflict:
                    continue
                # Score: prefer smaller differences (pack tighter)
                small_count = (new_bits & 0xFFFF).bit_count()
                candidates.append((small_count, x, new_bits))
            
            if not candidates:
                return None, 0
            
            # Selection
            if bias > 0 and rng.random() < bias:
                # Pick randomly from top candidates
                top_n = min(len(candidates), 10)
                idx = rng.randint(0, top_n - 1)
                _, x, new_bits = candidates[idx]
            else:
                # Pick best (smallest score)
                candidates.sort(key=lambda t: t[0])
                _, x, new_bits = candidates[0]
            
            row.append(x)
            local_bits |= new_bits
        
        return row, local_bits
    
    # Greedy initialization
    def greedy_init(seed):
        rng = random.Random(seed)
        used_bits = 0
        rows = []
        for _ in range(n_rows):
            row, bits = build_row_bitset(k, used_bits, 10000, rng, bias=0.3)
            if row is None:
                return None
            rows.append(row)
            used_bits |= bits
        return rows, used_bits
    
    # Get all difference bits for a row
    def row_diff_bits(row):
        bits = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                bits |= (1 << (row[j] - row[i]))
        return bits
    
    # Get max entry
    def get_max(rows):
        return max(max(r) for r in rows)
    
    # Find best greedy solution
    best_rows = None
    best_max = float('inf')
    best_used = 0
    
    for seed in range(30):
        if time_up():
            break
        res = greedy_init(seed)
        if res:
            rows, used = res
            m = get_max(rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]
                best_used = used
    
    if best_rows is None:
        # Fallback
        res = greedy_init(0)
        if res:
            best_rows, best_used = res
            best_max = get_max(best_rows)
        else:
            return []
    
    if best_max <= lb:
        return best_rows
    
    # Local search with simulated annealing
    rng = random.Random(123)
    current_rows = [r[:] for r in best_rows]
    current_used = best_used
    current_max = best_max
    
    # Track row maxes
    row_maxes = [max(r) for r in current_rows]
    
    no_improve = 0
    max_no_improve = 200
    temperature = 10.0
    cooling = 0.995
    
    while not time_up() and no_improve < max_no_improve:
        # Pick a row to rebuild: prefer rows that hold the current max
        max_rows = [i for i in range(n_rows) if row_maxes[i] == current_max]
        if max_rows:
            i = rng.choice(max_rows)
        else:
            i = rng.randint(0, n_rows - 1)
        
        old_row = current_rows[i]
        old_bits = row_diff_bits(old_row)
        old_max = max(old_row)
        
        # Remove from used
        # We can't simply XOR because bits might be shared? No, differences are unique across all rows.
        # So we can safely remove.
        current_used &= ~old_bits
        
        # Determine target max for rebuild
        if old_max == current_max:
            target_max = current_max - 1
        else:
            target_max = current_max  # reshuffle
        
        if target_max < k:
            target_max = k
        
        # Try to rebuild
        new_row, new_bits = build_row_bitset(k, current_used, target_max, rng, bias=0.5)
        
        if new_row is not None:
            new_max = max(new_row)
            
            # Simulated annealing acceptance
            if new_max < old_max:
                # Always accept improvement
                current_rows[i] = new_row
                current_used |= new_bits
                row_maxes[i] = new_max
                new_global_max = max(row_maxes)
                if new_global_max < current_max:
                    current_max = new_global_max
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Accept with probability based on temperature
                delta = new_max - old_max
                if delta == 0:
                    # Reshuffle: accept with moderate probability
                    if rng.random() < 0.3:
                        current_rows[i] = new_row
                        current_used |= new_bits
                        no_improve += 1
                    else:
                        current_used |= old_bits
                else:
                    # Worse: accept with SA probability
                    prob = 1.0 / (1.0 + delta * delta / max(temperature, 0.1))
                    if rng.random() < prob:
                        current_rows[i] = new_row
                        current_used |= new_bits
                        row_maxes[i] = new_max
                        no_improve += 1
                    else:
                        current_used |= old_bits
        else:
            # Failed to rebuild, restore
            current_used |= old_bits
            no_improve += 1
        
        # Cool down
        temperature *= cooling
        if temperature < 0.1:
            temperature = 0.1
    
    # Final validation
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
        return True
    
    if not validate(current_rows, n_rows, k):
        return best_rows
    
    return current_rows

# EVOLVE-BLOCK-END
