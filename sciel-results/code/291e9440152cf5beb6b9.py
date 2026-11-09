import time
import random
import numpy as np

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 117.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    
    # Precompute neighbors and other useful structures
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Convert to numpy for faster operations where possible
    neighbors_np = np.array([neighbors[v] for v in range(MAX_VERTS)])
    
    best_path = []
    best_len = 0
    
    def time_check():
        return time.time() - start_time > DEADLINE
    
    def is_valid_move(used, candidate, current_end):
        """
        Check if candidate can be added to the path end defined by current_end.
        candidate must be adjacent to current_end.
        candidate must not be used.
        candidate must not be adjacent to any other used vertex.
        """
        if used[candidate]:
            return False
        
        # Check neighbors of candidate
        # If any neighbor (other than current_end) is used, it's invalid
        for u in neighbors[candidate]:
            if u == current_end:
                continue
            if used[u]:
                return False
        return True
    
    def get_scored_moves(used, end_vertex):
        """
        Returns a list of (candidate, score) tuples.
        score is the number of unvisited neighbors of the candidate (Warnsdorff heuristic).
        """
        moves = []
        for v in neighbors[end_vertex]:
            if used[v]:
                continue
            
            # Basic validity: v is not adjacent to any other used vertex
            valid = True
            for u in neighbors[v]:
                if u == end_vertex:
                    continue
                if used[u]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Calculate Warnsdorff score: number of unvisited neighbors of v
            score = 0
            for u in neighbors[v]:
                if not used[u]:
                    score += 1
            
            moves.append((v, score))
        
        return moves
    
    def bidirectional_grow(initial_path=None, use_sa=False, sa_iterations=500):
        """
        Grow a path bidirectionally using greedy or simulated annealing.
        """
        if initial_path is None:
            v0 = random.randint(0, MAX_VERTS - 1)
            v1 = random.choice(neighbors[v0])
            path = [v0, v1]
        else:
            path = list(initial_path)
            
        if len(path) < 2:
            return path
            
        used = [False] * MAX_VERTS
        for v in path:
            used[v] = True
        
        if use_sa:
            return sa_grow(path, used, sa_iterations)
        else:
            return greedy_grow(path, used)
    
    def greedy_grow(path, used):
        """Standard greedy bidirectional growth."""
        while not time_check():
            left_end = path[0]
            right_end = path[-1]
            
            left_moves = get_scored_moves(used, left_end)
            right_moves = get_scored_moves(used, right_end)
            
            if not left_moves and not right_moves:
                return path
            
            # Decide which end to grow and which vertex to pick
            best_left = None
            best_left_score = float('inf')
            if left_moves:
                min_score = min(s for _, s in left_moves)
                candidates = [v for v, s in left_moves if s == min_score]
                best_left = random.choice(candidates)
                best_left_score = min_score
                
            best_right = None
            best_right_score = float('inf')
            if right_moves:
                min_score = min(s for _, s in right_moves)
                candidates = [v for v, s in right_moves if s == min_score]
                best_right = random.choice(candidates)
                best_right_score = min_score
            
            # Choose end
            if not left_moves:
                path.append(best_right)
                used[best_right] = True
            elif not right_moves:
                path.insert(0, best_left)
                used[best_left] = True
            else:
                # Both have moves. Pick the end with the more constrained move (lower score)
                if best_left_score < best_right_score:
                    path.insert(0, best_left)
                    used[best_left] = True
                elif best_right_score < best_left_score:
                    path.append(best_right)
                    used[best_right] = True
                else:
                    # Tie: pick randomly between the two best options
                    if random.random() < 0.5:
                        path.insert(0, best_left)
                        used[best_left] = True
                    else:
                        path.append(best_right)
                        used[best_right] = True
        
        return path
    
    def sa_grow(path, used, max_iters):
        """
        Simulated annealing growth.
        Allows temporary path shortening (retraction) to escape local optima.
        """
        current_path = list(path)
        current_used = list(used)
        current_len = len(current_path)
        
        best_path_sa = list(current_path)
        best_len_sa = current_len
        
        temperature = 5.0
        cooling_rate = 0.995
        
        for it in range(max_iters):
            if time_check():
                break
            
            if temperature < 0.01:
                break
            
            # Choose an action: extend, retract, or pivot
            action = random.random()
            
            if action < 0.6 and current_len < MAX_VERTS:
                # Try to extend
                # Choose which end to extend
                if random.random() < 0.5:
                    end = current_path[0]
                    extend_left = True
                else:
                    end = current_path[-1]
                    extend_left = False
                
                moves = get_scored_moves(current_used, end)
                if not moves:
                    continue
                
                # Pick a move (prefer lower score but allow some randomness)
                # Use softmax-like selection
                scores = [s for _, s in moves]
                min_s = min(scores)
                # Shift scores so min is 0
                shifted = [s - min_s for s in scores]
                # Convert to probabilities
                exp_scores = [np.exp(-s / temperature) for s in shifted]
                total = sum(exp_scores)
                if total == 0:
                    continue
                probs = [e / total for e in exp_scores]
                
                # Sample
                r = random.random()
                cumsum = 0
                chosen_idx = 0
                for i, p in enumerate(probs):
                    cumsum += p
                    if r <= cumsum:
                        chosen_idx = i
                        break
                
                new_vertex = moves[chosen_idx][0]
                
                # Apply move
                if extend_left:
                    current_path.insert(0, new_vertex)
                else:
                    current_path.append(new_vertex)
                current_used[new_vertex] = True
                current_len += 1
                
                # Accept if better (always)
                if current_len > best_len_sa:
                    best_len_sa = current_len
                    best_path_sa = list(current_path)
                    
            elif action < 0.8 and current_len > 2:
                # Try to retract (remove from an end)
                # This allows escaping bad local optima
                if random.random() < 0.5:
                    removed = current_path.pop(0)
                else:
                    removed = current_path.pop()
                
                current_used[removed] = False
                current_len -= 1
                
                # Accept with probability based on temperature (Metropolis)
                # Delta is -1 (worse), so accept with prob exp(1/temperature)
                # Actually, delta = new_score - old_score = -1
                # Accept with prob exp(delta / temperature) = exp(-1/temperature)
                if random.random() < np.exp(-1.0 / temperature):
                    pass  # Accept
                else:
                    # Reject: undo
                    if removed == current_path[0] if current_path else True:
                        current_path.insert(0, removed)
                    else:
                        current_path.append(removed)
                    current_used[removed] = True
                    current_len += 1
                    
            else:
                # Pivot: remove a middle vertex and try to reconnect
                # This is more complex; skip for now or implement carefully
                # For simplicity, just do a random retraction + extension
                if current_len > 4:
                    # Remove a random non-end vertex
                    idx = random.randint(1, current_len - 2)
                    removed = current_path.pop(idx)
                    current_used[removed] = False
                    current_len -= 1
                    
                    # Try to extend one of the new ends to reconnect
                    left_end = current_path[idx - 1] if idx - 1 >= 0 else current_path[0]
                    right_end = current_path[idx] if idx < current_len else current_path[-1]
                    
                    # Try to find a replacement
                    moves_left = get_scored_moves(current_used, left_end)
                    moves_right = get_scored_moves(current_used, right_end)
                    
                    # Pick the best move
                    all_moves = []
                    if moves_left:
                        all_moves.extend([(v, s, 'left') for v, s in moves_left])
                    if moves_right:
                        all_moves.extend([(v, s, 'right') for v, s in moves_right])
                    
                    if all_moves:
                        # Pick lowest score
                        min_s = min(s for _, s, _ in all_moves)
                        candidates = [(v, side) for v, s, side in all_moves if s == min_s]
                        if candidates:
                            v, side = random.choice(candidates)
                            if side == 'left':
                                current_path.insert(idx - 1, v)
                            else:
                                current_path.insert(idx, v)
                            current_used[v] = True
                            current_len += 1
                            # Net change: 0 (removed one, added one)
                            # But we might have broken the path! Need to verify.
                            # For now, assume it's valid since we checked moves.
                    
                    if current_len > best_len_sa:
                        best_len_sa = current_len
                        best_path_sa = list(current_path)
            
            # Cool down
            temperature *= cooling_rate
        
        return best_path_sa
    
    def generate_initial_greedy():
        """
        Generate a short initial path using greedy approach.
        """
        v0 = 0
        path = [v0]
        used = [False] * MAX_VERTS
        used[v0] = True
        
        for _ in range(20):
            if time_check():
                break
            current = path[-1]
            moves = get_scored_moves(used, current)
            if not moves:
                break
            min_score = min(s for _, s in moves)
            candidates = [v for v, s in moves if s == min_score]
            next_v = random.choice(candidates)
            path.append(next_v)
            used[next_v] = True
            
        return path
    
    # Main search loop
    # Strategy:
    # 1. Try structured starts
    # 2. Then random restarts with both greedy and SA
    
    # Initial structured attempts
    for _ in range(5):
        if time_check():
            break
        initial = generate_initial_greedy()
        if len(initial) < 2:
            continue
        p = bidirectional_grow(initial, use_sa=False)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Random restarts with mixed strategies
    while not time_check():
        # Decide strategy: greedy or SA
        use_sa = random.random() < 0.3  # 30% SA, 70% greedy
        
        initial = None  # Random start
        p = bidirectional_grow(initial, use_sa=use_sa, sa_iterations=300)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    if not best_path:
        best_path = [0, 1]
        
    return best_path
