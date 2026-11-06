import time
import random
import math
import bisect

def _score(S_bits):
    """
    S_bits is a bitset integer where bit i is 1 if i is in S.
    Returns count of distinct d > 0 such that a, a+d, a+2d in S.
    """
    count = 0
    for d in range(1, 256):
        if (S_bits & (S_bits >> d) & (S_bits >> (2 * d))) != 0:
            count += 1
    return count

def _generate_heuristic_moves(current_list, current_bits, MAXV, num_moves=5):
    """
    Generate moves that complete 3-term progressions.
    For each pair (a, b) in the set, try to add c = 2b - a or c = 2a - b.
    Also try to complete from the other side.
    Returns list of (x_to_remove, y_to_add) tuples.
    """
    moves = []
    N = len(current_list)
    
    # Get all elements for quick lookup
    in_set = set(current_list)
    
    # For each pair, generate completion candidates
    for i in range(N):
        a = current_list[i]
        for j in range(i + 1, N):
            b = current_list[j]
            d = b - a
            
            # Try a, b, a+2d (i.e., 2b - a)
            c1 = 2 * b - a
            if 1 <= c1 <= MAXV and c1 not in in_set:
                # Remove some random element, add c1
                moves.append((random.choice(current_list), c1))
            
            # Try b, a, 2a-b (i.e., a - d)
            c2 = a - d
            if 1 <= c2 <= MAXV and c2 not in in_set:
                moves.append((random.choice(current_list), c2))
    
    # Shuffle and take up to num_moves
    random.shuffle(moves)
    return moves[:num_moves]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    N = 32
    MAXV = 512
    
    best_score = -1
    best_list = []
    
    # Run multiple restarts if time permits
    while time.time() < deadline:
        # Initialize random set
        current_list = random.sample(range(1, MAXV + 1), N)
        current_list.sort()
        current_bits = 0
        for x in current_list:
            current_bits |= (1 << x)
        
        current_score = _score(current_bits)
        
        if current_score > best_score:
            best_score = current_score
            best_list = current_list.copy()
        
        # Simulated Annealing
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        
        # Local time limit for this restart
        local_deadline = start_time + min(8.0, deadline - start_time)
        
        while time.time() < local_deadline:
            # Mix heuristic and random moves
            use_heuristic = random.random() < 0.4
            
            if use_heuristic:
                heuristic_moves = _generate_heuristic_moves(current_list, current_bits, MAXV, num_moves=3)
                if heuristic_moves:
                    x, y = random.choice(heuristic_moves)
                else:
                    use_heuristic = False
            
            if not use_heuristic:
                # Pick x to remove
                x_idx = random.randint(0, N - 1)
                x = current_list[x_idx]
                
                # Pick y to add
                while True:
                    y = random.randint(1, MAXV)
                    if not ((current_bits >> y) & 1):
                        break
            
            # Check if x is still in the set (it should be, but safety)
            if x not in current_list:
                x_idx = random.randint(0, N - 1)
                x = current_list[x_idx]
            
            # Update bits
            new_bits = current_bits ^ (1 << x) ^ (1 << y)
            
            # Update list
            x_idx = current_list.index(x)
            new_list = current_list[:x_idx] + current_list[x_idx+1:]
            ins_idx = bisect.bisect_left(new_list, y)
            new_list.insert(ins_idx, y)
            
            new_score = _score(new_bits)
            
            delta = new_score - current_score
            
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                current_list = new_list
                current_bits = new_bits
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_list = current_list.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min

    return best_list
