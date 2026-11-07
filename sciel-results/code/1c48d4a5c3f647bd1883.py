import random
import time
import math

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

# Precompute adjacency matrix for fast checks
N = P['n']
adj = [[False] * N for _ in range(N)]
for u, v in P['edges']:
    adj[u][v] = True
    adj[v][u] = True

def is_clique(part):
    for i in range(len(part)):
        for j in range(i + 1, len(part)):
            if not adj[part[i]][part[j]]:
                return False
    return True

def is_independent(part):
    for i in range(len(part)):
        for j in range(i + 1, len(part)):
            if adj[part[i]][part[j]]:
                return False
    return True

def is_valid_part(part):
    return is_clique(part) or is_independent(part)

def is_valid_partition(parts):
    # Check if every part is valid
    for part in parts:
        if not is_valid_part(part):
            return False
    # Check if all vertices are covered exactly once
    all_vertices = []
    for part in parts:
        all_vertices.extend(part)
    if sorted(all_vertices) != list(range(N)):
        return False
    return True

def greedy_partition():
    """Initial greedy solution from seed logic."""
    es = {tuple(e) for e in P['edges']}
    es.add(tuple(reversed(e)) for e in P['edges']) # Ensure both directions? No, set of tuples needs both or check both ways.
    # Actually, is_clique uses adj matrix, so we don't need es for validity checks in the new code.
    # But for the initial greedy, we can just use a simple heuristic.
    parts = []
    for v in range(N):
        placed = False
        for part in parts:
            trial = part + [v]
            if is_valid_part(trial):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts

def generate_neighbor(parts):
    """Generate a random valid neighbor of the current partition."""
    parts = [list(p) for p in parts] # Deep copy
    num_parts = len(parts)
    
    # Choose a move type
    move_type = random.random()
    
    if move_type < 0.7:
        # Move a vertex to an existing part
        if num_parts == 1:
            return None # Cannot move if only 1 part
        
        src_idx = random.randint(0, num_parts - 1)
        if len(parts[src_idx]) <= 1:
            return None
        
        v = random.choice(parts[src_idx])
        parts[src_idx].remove(v)
        
        # Pick a target part
        target_candidates = [i for i in range(num_parts) if i != src_idx]
        if not target_candidates:
            parts[src_idx].append(v)
            return None
            
        dst_idx = random.choice(target_candidates)
        
        # Check validity
        trial = parts[dst_idx] + [v]
        if is_valid_part(trial):
            parts[dst_idx].append(v)
            new_parts = [p for p in parts if p]
            return new_parts
        else:
            parts[src_idx].append(v)
            return None

    elif move_type < 0.9:
        # Merge two parts
        if len(parts) < 2:
            return None
            
        i, j = random.sample(range(len(parts)), 2)
        
        trial = parts[i] + parts[j]
        if is_valid_part(trial):
            parts[i].extend(parts[j])
            parts.pop(j)
            return parts
        else:
            return None

    else:
        # Split (less frequent, only to escape local minima)
        if not parts:
            return None
        
        src_idx = random.randint(0, len(parts) - 1)
        if len(parts[src_idx]) <= 1:
            return None
            
        v = random.choice(parts[src_idx])
        parts[src_idx].remove(v)
        
        new_parts = [p for p in parts if p]
        new_parts.append([v])
        return new_parts

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin for 60s limit
    
    # Initial solution
    current = greedy_partition()
    best = [list(p) for p in current]
    best_cost = len(best)
    
    # Simulated Annealing
    T = 5.0
    cooling_rate = 0.99
    
    while time.time() < deadline:
        neighbor = generate_neighbor(current)
        if neighbor is None:
            continue
            
        current_cost = len(neighbor)
        delta_cost = current_cost - len(current)
        
        if delta_cost < 0:
            # Better
            current = neighbor
            if current_cost < best_cost:
                best = [list(p) for p in current]
                best_cost = current_cost
        else:
            # Worse or equal
            if delta_cost == 0:
                # Accept with some probability to explore
                if random.random() < 0.5:
                    current = neighbor
            else:
                # Accept worse with probability exp(-delta/T)
                prob = math.exp(-delta_cost / T)
                if random.random() < prob:
                    current = neighbor
                    
        T *= cooling_rate
        if T < 0.1:
            T = 5.0 # Reheat

    # Final Polish: Try to improve best solution with simple local moves
    # We work on a copy of best
    polish_parts = [list(p) for p in best]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try moving each vertex to any other part
        for i in range(len(polish_parts)):
            if len(polish_parts[i]) <= 1:
                continue
            v = polish_parts[i][0]
            # Try moving v to every other part
            for j in range(len(polish_parts)):
                if i == j:
                    continue
                # Check if v can be added to j
                trial = polish_parts[j] + [v]
                if is_valid_part(trial):
                    # Check if removing v from i leaves i valid (it always does, subset of valid part? No, removing vertex from clique/ind-set keeps it valid)
                    # Actually, if i was a clique, removing a vertex keeps it a clique. If ind-set, removing keeps it ind-set.
                    # So we just need to check if v fits in j.
                    # Also need to check if i becomes empty.
                    new_i = [x for x in polish_parts[i] if x != v]
                    if not new_i:
                        # If i becomes empty, we reduce part count by 1
                        # This is a good move
                        polish_parts[i] = []
                        polish_parts[j].append(v)
                        # Remove empty parts
                        polish_parts = [p for p in polish_parts if p]
                        improved = True
                        break
                    else:
                        # Part count stays same, but we might enable future merges?
                        # For now, let's only accept if it reduces count or keeps it same but we are polishing.
                        # Since we are polishing for min count, we only accept if count decreases.
                        # Wait, if count stays same, it's not an improvement in cost.
                        # So we only accept if new_i is empty.
                        pass
            if improved:
                break
        
        # Try merging any two parts
        for i in range(len(polish_parts)):
            for j in range(i + 1, len(polish_parts)):
                trial = polish_parts[i] + polish_parts[j]
                if is_valid_part(trial):
                    polish_parts[i] = trial
                    polish_parts[j] = []
                    polish_parts = [p for p in polish_parts if p]
                    improved = True
                    break
            if improved:
                break
    
    if len(polish_parts) < best_cost:
        best = polish_parts
        best_cost = len(best)

    # Ensure types are correct: list of lists of ints
    # Convert to tuples if necessary? Task says "list or tuple". List is fine.
    # Integers must be Python int. They are.
    
    # Final check
    if not is_valid_partition(best):
        # Fallback to greedy if something went wrong (shouldn't happen)
        best = greedy_partition()
        
    return best
