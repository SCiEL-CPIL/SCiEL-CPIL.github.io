"""Improved snake-in-the-box for Q_9 using Beam Search with optimized state."""
import sys
import time

N = 9
FULL_MASK = (1 << N) - 1

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a long induced path (snake) in Q_9 using Beam Search.
    Uses an optimized state representation to allow for larger beam sizes
    and faster validity checks.
    """
    start_time = time.time()
    time_limit = 115  # Leave buffer
    
    # Beam size
    beam_size = 200 
    
    # Precompute neighbors
    neighbors = [[] for _ in range(1 << N)]
    for v in range(1 << N):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Initial state
    start_vertex = 0
    initial_path = [start_vertex]
    initial_used = {start_vertex}
    # blocked_set: vertices that cannot be the next one because they are adjacent to path[:-1]
    # Initially path has length 1, so path[:-1] is empty. blocked_set is empty.
    initial_blocked = set()
    
    # Beam contains tuples: (path, used, blocked_set, score)
    # We store path to reconstruct, used for check, blocked for check.
    # Score is for sorting.
    
    def get_score(path, blocked_set):
        """
        Heuristic: Prefer longer paths.
        Tie-breaker: Number of available moves from the end.
        """
        end = path[-1]
        # Available moves are neighbors of end that are not in used and not in blocked_set
        # We don't have 'used' in the score function signature easily without passing it,
        # but we can pass the state.
        # Actually, let's just use length for primary sort, and maybe a random tie-breaker
        # or a simple heuristic like degree.
        # To keep it fast, let's just use length.
        # If we want better pruning, we can count available moves.
        # Let's pass used as well.
        pass # Placeholder

    # Let's integrate score calculation into the expansion loop for efficiency.
    
    beam = [(initial_path, initial_used, initial_blocked)]
    
    best_path = initial_path
    best_len = 1
    
    while time.time() - start_time < time_limit:
        next_beam = []
        
        for path, used, blocked_set in beam:
            if time.time() - start_time > time_limit:
                break
                
            cur_end = path[-1]
            
            for v in neighbors[cur_end]:
                # Check validity
                if v in used:
                    continue
                if v in blocked_set:
                    continue
                
                # Valid extension
                new_path = path + [v]
                new_used = used | {v}
                
                # Update blocked_set
                # The new blocked set for the NEXT step must contain neighbors of path[0]...path[-1]
                # Old blocked_set contained neighbors of path[0]...path[-2]
                # We need to add neighbors of path[-1] (which is cur_end)
                # Note: cur_end is already in 'used', so its neighbors are "blocked" for future use?
                # Yes, if a future vertex is adjacent to cur_end, it would form a chord with cur_end.
                # Since cur_end is in the path (not at the very end of the *new* path's "previous" part? 
                # Wait.
                # Path: v0 ... vk.
                # Next vertex v_{k+1} must not be adjacent to v0 ... v_{k-1}.
                # It CAN be adjacent to vk (it is, by definition).
                # So, for the next step, the "forbidden" vertices are neighbors of v0 ... vk.
                # The current `blocked_set` corresponds to neighbors of v0 ... v_{k-1}.
                # So we add neighbors of vk (cur_end).
                
                new_blocked_set = blocked_set | set(neighbors[cur_end])
                
                # Update best
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = new_path[:]
                    
                # Calculate score for sorting
                # We want to keep paths that have more future options.
                # Count available moves from v
                # Available moves are neighbors of v not in new_used and not in new_blocked_set
                # Note: new_blocked_set is the set of vertices forbidden for the step AFTER adding v.
                # So if w is a neighbor of v, and w is not in new_used and w is not in new_blocked_set,
                # then w is a valid next move.
                
                # Optimization: We can just count how many neighbors of v are NOT in new_blocked_set
                # and NOT in new_used.
                # Since new_used contains v, and v is not its own neighbor, we just check new_blocked_set
                # and the rest of new_used?
                # Actually, if w is in new_used, it's invalid.
                # If w is in new_blocked_set, it's invalid.
                
                # To keep it fast, let's just use length as the primary key.
                # Tie-breaking with a simple heuristic:
                # Number of neighbors of v not in new_blocked_set.
                # (We ignore new_used for the heuristic to save time, as it's a small set compared to full space,
                #  but actually new_used is small too. Let's just check new_blocked_set for speed).
                
                # A better heuristic: 
                # Score = (len(new_path), count_valid_next(v, new_used, new_blocked_set))
                
                # Let's implement a fast count:
                # valid_next_count = 0
                # for w in neighbors[v]:
                #     if w not in new_used and w not in new_blocked_set:
                #         valid_next_count += 1
                
                # This is O(N) per candidate. 9 checks. Very fast.
                
                valid_next_count = 0
                for w in neighbors[v]:
                    if w not in new_used and w not in new_blocked_set:
                        valid_next_count += 1
                
                # Score tuple: (Length, ValidNextCount)
                # We sort descending.
                score = (len(new_path), valid_next_count)
                
                next_beam.append((new_path, new_used, new_blocked_set, score))
        
        if not next_beam:
            break
            
        # Sort by score descending
        next_beam.sort(key=lambda x: x[3], reverse=True)
        
        # Keep top beam_size
        # We should also try to deduplicate if possible, but paths are sequences, so unique.
        beam = next_beam[:beam_size]
        
        # If the best score in the beam is stuck (same length for many iterations),
        # we might want to increase randomness or change strategy.
        # For now, standard beam search.

    return best_path
# EVOLVE-BLOCK-END
