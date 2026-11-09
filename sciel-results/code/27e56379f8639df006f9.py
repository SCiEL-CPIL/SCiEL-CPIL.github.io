import time
import random

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0  # Safety margin for 120s limit

def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    
    Strategy:
    1. Bitmask representation for O(1) validity checks.
    2. Bidirectional growth with Warnsdorff's heuristic.
    3. Multi-restart with varied initial seeds and randomization.
    
    The bitmask approach allows checking if a vertex is adjacent to any used vertex
    in constant time, enabling thousands of restarts within the time limit.
    """
    start_time = time.time()
    
    # Precompute neighbors and neighbor masks for all vertices
    neighbors = [[] for _ in range(MAX_VERTS)]
    neighbor_masks = [0] * MAX_VERTS
    for v in range(MAX_VERTS):
        mask = 0
        for b in range(N):
            nb = v ^ (1 << b)
            neighbors[v].append(nb)
            mask |= (1 << nb)
        neighbor_masks[v] = mask

    best_path = []
    best_len = 0
    
    def get_valid_moves(used_mask, end_vertex):
        """
        Returns list of valid candidates to extend from end_vertex.
        A candidate v is valid if:
        1. v is adjacent to end_vertex
        2. v is not used
        3. v is not adjacent to any other used vertex (except end_vertex)
        """
        moves = []
        for v in neighbors[end_vertex]:
            if used_mask & (1 << v):
                continue
            # Check if v is adjacent to any used vertex other than end_vertex
            # Used vertices excluding end_vertex is (used_mask ^ (1 << end_vertex))
            if neighbor_masks[v] & (used_mask ^ (1 << end_vertex)):
                continue
            moves.append(v)
        return moves

    def score_move(used_mask, v, end_vertex):
        """
        Score a move. Lower is better (more constrained).
        We use a 1-step lookahead: simulate adding v, then count valid moves from v.
        """
        # Simulate adding v
        new_used_mask = used_mask | (1 << v)
        
        # Count valid moves from v (new end)
        count = 0
        for u in neighbors[v]:
            if new_used_mask & (1 << u):
                continue
            # Check if u is adjacent to any used vertex other than v
            # Used vertices excluding v is (new_used_mask ^ (1 << v))
            if neighbor_masks[u] & (new_used_mask ^ (1 << v)):
                continue
            count += 1
            
        return count

    def bidirectional_grow(initial_path, rng, time_limit):
        """
        Grow a snake bidirectionally from an initial path using Warnsdorff's rule.
        """
        if not initial_path or len(initial_path) < 2:
            return initial_path

        path = list(initial_path)
        used_mask = 0
        for v in path:
            used_mask |= (1 << v)
        
        while True:
            if time.time() > time_limit:
                break
            
            left_end = path[0]
            right_end = path[-1]
            
            left_moves = get_valid_moves(used_mask, left_end)
            right_moves = get_valid_moves(used_mask, right_end)
            
            if not left_moves and not right_moves:
                break
            
            # Score all moves to find the most constrained (Warnsdorff)
            best_left = None
            best_left_score = float('inf')
            if left_moves:
                # To speed up, we can sample if the list is large, but usually it's small
                for v in left_moves:
                    s = score_move(used_mask, v, left_end)
                    if s < best_left_score:
                        best_left_score = s
                        best_left = v
            
            best_right = None
            best_right_score = float('inf')
            if right_moves:
                for v in right_moves:
                    s = score_move(used_mask, v, right_end)
                    if s < best_right_score:
                        best_right_score = s
                        best_right = v
            
            # Choose which end to extend
            # Priority: Extend the side with fewer options (more constrained)
            # If one side is blocked, extend the other
            if not left_moves:
                if best_right is not None:
                    path.append(best_right)
                    used_mask |= (1 << best_right)
                else:
                    break
            elif not right_moves:
                if best_left is not None:
                    path.insert(0, best_left)
                    used_mask |= (1 << best_left)
                else:
                    break
            else:
                # Both sides have moves
                # Pick the side with the lower score (more constrained)
                if best_left_score < best_right_score:
                    path.insert(0, best_left)
                    used_mask |= (1 << best_left)
                elif best_right_score < best_left_score:
                    path.append(best_right)
                    used_mask |= (1 << best_right)
                else:
                    # Tie-breaking: Random choice to diversify
                    if rng.random() < 0.5:
                        path.insert(0, best_left)
                        used_mask |= (1 << best_left)
                    else:
                        path.append(best_right)
                        used_mask |= (1 << best_right)
                        
        return path

    def generate_initial_seed(rng, length_target=10):
        """
        Generate a short, valid initial path to serve as a seed for bidirectional growth.
        """
        v0 = rng.randint(0, MAX_VERTS - 1)
        path = [v0]
        used_mask = 1 << v0
        
        for _ in range(length_target):
            current = path[-1]
            moves = get_valid_moves(used_mask, current)
            if not moves:
                break
            
            # Greedy choice for initialization: pick the one with lowest score
            min_score = float('inf')
            best_v = None
            for v in moves:
                # Quick score: just count unvisited neighbors for speed in init
                s = 0
                for u in neighbors[v]:
                    if not (used_mask & (1 << u)):
                        s += 1
                if s < min_score:
                    min_score = s
                    best_v = v
            
            if best_v is None:
                break
                
            path.append(best_v)
            used_mask |= (1 << best_v)
            
        return path

    # Multi-restart strategy
    # We use a global RNG for reproducibility if needed, but random is fine
    main_rng = random.Random(42)
    
    # Phase 1: Structured restarts with different seeds
    # We run a fixed number of high-quality attempts first
    max_structured_attempts = 50
    
    for i in range(max_structured_attempts):
        if time.time() > DEADLINE - 20: # Reserve time for random restarts
            break
            
        # Create a new RNG for this attempt to ensure independence
        attempt_rng = random.Random(i * 1000 + 7)
        
        # Generate a short seed
        seed_path = generate_initial_seed(attempt_rng, length_target=8)
        
        if len(seed_path) < 2:
            continue
            
        # Grow bidirectionally
        # Time limit for this specific attempt: leave enough time for the loop check
        time_limit = DEADLINE - (time.time() - start_time) * 0.1 # Dynamic? No, just use global deadline check inside
        # Actually, bidirectional_grow checks global time. We just pass DEADLINE.
        p = bidirectional_grow(seed_path, attempt_rng, DEADLINE)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
            
    # Phase 2: Random restarts until time runs out
    # Use the main RNG for this phase
    while time.time() < DEADLINE:
        # Generate a random seed of length 2
        v0 = main_rng.randint(0, MAX_VERTS - 1)
        v1 = main_rng.choice(neighbors[v0])
        seed_path = [v0, v1]
        
        # Optionally extend the seed a bit more for better starting point
        # This helps avoid immediate dead ends
        used_mask = (1 << v0) | (1 << v1)
        extend_steps = main_rng.randint(0, 5)
        for _ in range(extend_steps):
            current = seed_path[-1]
            moves = get_valid_moves(used_mask, current)
            if not moves:
                break
            # Random choice for diversity in seeds
            next_v = main_rng.choice(moves)
            seed_path.append(next_v)
            used_mask |= (1 << next_v)
            
        p = bidirectional_grow(seed_path, main_rng, DEADLINE)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    # Fallback if something went wrong (should not happen)
    if not best_path:
        best_path = [0, 1]
        
    return best_path

# Entry point required by the evaluator
# The task specifies: construct() -> list of distinct integers
# We have defined construct() above.

if __name__ == "__main__":
    # For local testing
    path = construct()
    print(f"Found snake of length: {len(path)}")
    # Validate
    valid = True
    for i in range(len(path)):
        for j in range(len(path)):
            if abs(i-j) > 1:
                if (path[i] ^ path[j]).bit_count() == 1:
                    valid = False
                    break
        if not valid:
            break
    for i in range(len(path)-1):
        if (path[i] ^ path[i+1]).bit_count() != 1:
            valid = False
            break
    print(f"Valid: {valid}")
