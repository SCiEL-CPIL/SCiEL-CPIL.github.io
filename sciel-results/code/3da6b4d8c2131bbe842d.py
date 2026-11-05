import time
import random
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1"
    
    # Standard recursive construction for base cases
    def standard_superpermutation(k):
        if k == 1:
            return [1]
        word = standard_superpermutation(k - 1)
        perms, seen = [], set()
        for i in range(len(word) - (k - 1) + 1):
            window = tuple(word[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [k] + list(p)
            # Find overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        return new

    if n <= 6:
        word = standard_superpermutation(n)
        return "".join(map(str, word))

    if n == 7:
        # Get standard W_6
        w6 = standard_superpermutation(6)
        
        # Extract order of permutations of 1..6 from W_6
        perms, seen = [], set()
        for i in range(len(w6) - 5):
            window = tuple(w6[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Initial order is the standard one
        current_order = perms[:]
        
        # Precompute all valid permutations of 1..7 for validation
        all_perms_7 = set("".join(map(str, p)) for p in permutations(range(1, 8)))
        
        def build_word(order):
            new = []
            for p in order:
                block = list(p) + [7] + list(p)
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            return new

        def is_valid(word):
            s = "".join(map(str, word))
            for i in range(len(s) - 6):
                if s[i:i+7] not in all_perms_7:
                    return False
            return True

        best_word = build_word(current_order)
        best_len = len(best_word)
        
        start_time = time.time()
        deadline = start_time + 110  # Safety margin
        
        # Local Search
        # Moves: swap adjacent, 2-opt (reverse segment)
        
        while time.time() < deadline:
            # Pick a random move
            move_type = random.randint(0, 1)
            new_order = current_order[:]
            
            if move_type == 0:
                # Swap adjacent
                i = random.randint(0, len(new_order) - 2)
                new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            else:
                # 2-opt: reverse a segment
                i = random.randint(0, len(new_order) - 2)
                j = random.randint(i + 1, len(new_order) - 1)
                new_order[i:j+1] = new_order[i:j+1][::-1]
                
            # Build and check
            candidate_word = build_word(new_order)
            candidate_len = len(candidate_word)
            
            if candidate_len < best_len:
                if is_valid(candidate_word):
                    best_word = candidate_word
                    best_len = candidate_len
                    current_order = new_order
                else:
                    # Invalid, revert
                    pass
            else:
                # If length is same or worse, we might still accept it if it's valid 
                # to allow escaping local minima, but strictly reducing is better for score.
                # However, if we don't accept equal-length valid moves, we might get stuck.
                # Let's accept valid moves with same length to explore the space.
                if candidate_len == best_len:
                    if is_valid(candidate_word):
                        # Accept with some probability or always?
                        # Always accept to explore
                        current_order = new_order
                        best_word = candidate_word
                    else:
                        pass
                else:
                    # Reject worse length
                    pass
                    
    return "".join(map(str, best_word))
