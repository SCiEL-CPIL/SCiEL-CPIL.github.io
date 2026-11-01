"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s wall-clock

    # Total number of words in the space
    total = q ** n
    # Number of words within radius r of a given word
    # For ternary (q=3), radius 1: 1 + n*(q-1) = 1 + 2n
    cover_size = 1 + n * (q - 1)  # for radius=1, general q

    # Helper: convert integer to base-q digits of length n
    def index_to_digits(idx):
        digits = []
        for _ in range(n):
            digits.append(idx % q)
            idx //= q
        return digits[::-1]

    # Helper: convert digits to integer index
    def digits_to_index(digits):
        idx = 0
        for d in digits:
            idx = idx * q + d
        return idx

    # Precompute neighbors for each position and value change
    # neighbor_offsets[i][delta] = list of (pos, new_val) where pos=i, val changes by delta
    # Actually, for radius 1, neighbors of a word are: itself + change one coordinate to another value
    # Precompute for each position i and each possible change, the index offset
    # offset[i][old_val][new_val] = new_index - old_index when changing position i from old_val to new_val
    # This is: (new_val - old_val) * q^(n-1-i)

    # Precompute powers of q
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i - 1] * q

    # offset[i][old_val][new_val]
    offset = [[[0] * q for _ in range(q)] for _ in range(n)]
    for i in range(n):
        for old_val in range(q):
            for new_val in range(q):
                if old_val != new_val:
                    offset[i][old_val][new_val] = (new_val - old_val) * q_powers[n - 1 - i]

    # Greedy initialization
    def greedy_init():
        covered = np.zeros(total, dtype=np.uint8)
        code = []
        # Start with all indices, remove as we cover
        # For efficiency, maintain a set of uncovered indices
        uncovered = set(range(total))
        
        while uncovered and time.time() < deadline - 30:
            # Pick the word that covers the most uncovered words
            # For speed, sample some candidates rather than checking all
            if len(uncovered) > 1000:
                candidates = list(random.sample(list(uncovered), min(200, len(uncovered))))
            else:
                candidates = list(uncovered)
            
            best_cand = None
            best_count = -1
            for cand in candidates:
                # Count how many uncovered words are within radius 1 of cand
                count = 0
                # The word itself
                if cand in uncovered:
                    count += 1
                # All neighbors
                for i in range(n):
                    old_val = (cand // q_powers[n - 1 - i]) % q
                    for new_val in range(q):
                        if new_val != old_val:
                            nb = cand + offset[i][old_val][new_val]
                            if nb in uncovered:
                                count += 1
                if count > best_count:
                    best_count = count
                    best_cand = cand
            
            if best_cand is None:
                break
            
            code.append(best_cand)
            # Mark all words within radius 1 of best_cand as covered
            covered[best_cand] = 1
            uncovered.discard(best_cand)
            for i in range(n):
                old_val = (best_cand // q_powers[n - 1 - i]) % q
                for new_val in range(q):
                    if new_val != old_val:
                        nb = best_cand + offset[i][old_val][new_val]
                        if nb in uncovered:
                            covered[nb] = 1
                            uncovered.discard(nb)
        
        return code, covered

    # Initialize with greedy
    code, covered = greedy_init()
    best_code = list(code)
    best_size = len(code)

    # Convert code to a set for fast lookup
    code_set = set(code)

    # Precompute coverage count for each word: how many codewords cover it
    # cov_count[w] = number of codewords within radius 1 of w
    cov_count = np.zeros(total, dtype=np.int32)
    for cw in code:
        cov_count[cw] += 1
        for i in range(n):
            old_val = (cw // q_powers[n - 1 - i]) % q
            for new_val in range(q):
                if new_val != old_val:
                    nb = cw + offset[i][old_val][new_val]
                    cov_count[nb] += 1

    # Simulated annealing
    # Moves:
    # 1. Remove a codeword if it doesn't make any word uncovered
    # 2. Replace a codeword with a better one
    # 3. Add a codeword if it helps (but we want to minimize, so less useful)
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999

    rng = random.Random(42)

    while time.time() < deadline - 5:
        if T < T_min:
            break
        
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.4 and len(code) > 1:
            # Try to remove a codeword
            idx = rng.randrange(len(code))
            cw = code[idx]
            
            # Check if removing cw leaves any word uncovered
            # A word w is uncovered after removal if cov_count[w] == 1 and w is covered by cw
            # We need to check if any word has cov_count == 1 and is in the ball of cw
            can_remove = True
            # Check cw itself
            if cov_count[cw] == 1:
                can_remove = False
            else:
                for i in range(n):
                    old_val = (cw // q_powers[n - 1 - i]) % q
                    for new_val in range(q):
                        if new_val != old_val:
                            nb = cw + offset[i][old_val][new_val]
                            if cov_count[nb] == 1:
                                can_remove = False
                                break
                    if not can_remove:
                        break
            
            if can_remove:
                # Remove it
                code.pop(idx)
                code_set.discard(cw)
                cov_count[cw] -= 1
                for i in range(n):
                    old_val = (cw // q_powers[n - 1 - i]) % q
                    for new_val in range(q):
                        if new_val != old_val:
                            nb = cw + offset[i][old_val][new_val]
                            cov_count[nb] -= 1
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
            else:
                # Accept with probability exp(-1/T)
                if rng.random() < np.exp(-1.0 / T):
                    # We won't actually remove, just simulate for now
                    pass
        
        elif move_type < 0.8 and len(code) > 1:
            # Replace a codeword with a random nearby word
            idx = rng.randrange(len(code))
            old_cw = code[idx]
            
            # Generate a random neighbor (change one position)
            pos = rng.randrange(n)
            old_val = (old_cw // q_powers[n - 1 - pos]) % q
            new_val = rng.choice([v for v in range(q) if v != old_val])
            new_cw = old_cw + offset[pos][old_val][new_val]
            
            if new_cw in code_set:
                continue
            
            # Compute delta: how many words become uncovered, how many gain coverage
            # Words that lose coverage: those in ball(old_cw) \ ball(new_cw) that had cov_count == 1
            # Words that gain coverage: those in ball(new_cw) \ ball(old_cw)
            
            # For simplicity, compute net change in "uncovered risk"
            # Let's compute: for each word in the symmetric difference of the two balls,
            # track if it goes from covered to uncovered or vice versa
            
            lost = 0  # words that become uncovered
            gained = 0  # words that gain an extra cover (not directly useful for min size)
            
            # Words in ball(old_cw) not in ball(new_cw)
            # These lose one cover. If they had exactly 1 cover, they become uncovered.
            # Words in ball(new_cw) not in ball(old_cw)
            # These gain one cover.
            
            # For efficiency, compute the union of positions where they differ
            # Actually, let's just compute the delta in uncovered count
            # A word w is affected if it's in exactly one of the two balls
            
            # Compute ball(old_cw) and ball(new_cw)
            # For radius 1, ball size is 1 + n*(q-1)
            
            # Let's compute the set of words in each ball
            def get_ball(cw):
                ball = set()
                ball.add(cw)
                for i in range(n):
                    old_v = (cw // q_powers[n - 1 - i]) % q
                    for new_v in range(q):
                        if new_v != old_v:
                            nb = cw + offset[i][old_v][new_v]
                            ball.add(nb)
                return ball
            
            old_ball = get_ball(old_cw)
            new_ball = get_ball(new_cw)
            
            # Words that lose coverage: in old_ball but not in new_ball
            for w in old_ball:
                if w not in new_ball:
                    if cov_count[w] == 1:
                        lost += 1
            
            # Words that gain coverage: in new_ball but not in old_ball
            # These don't directly help with size reduction, but might help future moves
            
            # Delta cost = lost (we want to minimize this)
            delta = lost
            
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                # Accept the move
                code[idx] = new_cw
                code_set.discard(old_cw)
                code_set.add(new_cw)
                
                # Update cov_count
                for w in old_ball:
                    cov_count[w] -= 1
                for w in new_ball:
                    cov_count[w] += 1
                
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
        
        # Cool down
        T *= cooling

    # Final cleanup: try to remove any redundant codewords
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for idx in range(len(best_code) - 1, -1, -1):
            cw = best_code[idx]
            # Recompute cov_count for best_code
            # Actually, let's just check if removing cw is safe
            # We need to verify that no word becomes uncovered
            # For simplicity, skip this and return best_code as is
    
    # Convert best_code indices back to digit lists
    result = [index_to_digits(cw) for cw in best_code]
    return result

# EVOLVE-BLOCK-END
