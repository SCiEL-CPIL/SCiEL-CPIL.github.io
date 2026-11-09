import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    N = 32
    MAX_VAL = 512
    
    # We want to maximize the number of distinct d > 0 such that a, a+d, a+2d in S.
    # This is equivalent to counting the number of 3-term arithmetic progressions (3-APs) in S.
    # A 3-AP is defined by (a, d) where a, a+d, a+2d are in S.
    # Note: The problem asks for the number of DISTINCT d.
    # Wait, re-reading the task: "The score is the number of distinct d > 0 such that a, a + d, a + 2d all lie in the set for some a."
    # This is NOT the total number of 3-APs. It is the size of the set of differences D = {d | exists a, a,a+d,a+2d in S}.
    # This is a harder objective to optimize incrementally than the count of 3-APs.
    # Let's re-verify the seed code.
    # Seed code:
    # def score_mask(mask):
    #     count = 0
    #     for d in range(1, 256):
    #         if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
    #             count += 1
    #     return count
    # Yes, it counts distinct d.
    
    # Optimizing "number of distinct d" incrementally is tricky because adding an element might complete a new d, 
    # but if that d was already present, the score doesn't increase. 
    # However, we can still use a faster evaluation than O(512) bitmask checks.
    
    # Alternative: We can precompute for each d, the set of a such that (a, a+d, a+2d) is a valid AP in 1..512.
    # For a given set S, d is "active" if there exists a in S such that a+d in S and a+2d in S.
    # Let Active_d be a boolean.
    # Score = sum(Active_d).
    
    # To update efficiently:
    # When adding x to S:
    #   For each d, if (x-d in S and x-2d in S) OR (x in S and x+d in S and x+2d in S) ... wait.
    #   The condition for d being active is: exists a such that {a, a+d, a+2d} subset of S.
    #   If we add x, new APs involving x can start at x, x-d, or x-2d.
    #   So for each d:
    #      Check if (x-d in S and x-2d in S) -> AP (x-2d, x-d, x)
    #      Check if (x+d in S and x+2d in S) -> AP (x, x+d, x+2d)
    #      Check if (x-d in S and x+d in S) -> AP (x-d, x, x+d)
    #   If any of these is true, d becomes active.
    #   We need to track which d are currently active.
    #   If d was inactive and now active, score +1.
    
    # When removing x from S:
    #   For each d, check if the only APs for d were those involving x.
    #   This is harder. We need to know the count of APs for each d.
    #   Let cnt_d be the number of APs with difference d in S.
    #   Active_d is (cnt_d > 0).
    #   Score = sum(1 for d if cnt_d > 0).
    
    # This approach requires maintaining cnt_d for all d in 1..255.
    # Updating cnt_d when adding x:
    #   For each d in 1..255:
    #      if (x-d in S and x-2d in S): cnt_d += 1
    #      if (x+d in S and x+2d in S): cnt_d += 1
    #      if (x-d in S and x+d in S): cnt_d += 1
    #   This is O(255) per addition.
    #   Total for 32 elements: 32 * 255 * 3 ~ 25k ops. Very fast.
    
    # Updating cnt_d when removing x:
    #   Same checks, subtract 1.
    
    # This allows O(255) evaluation of a move (add/remove), which is ~100x faster than O(512) bitmask shifts+ands 
    # if implemented in Python? Bitmask shifts are fast in C, but Python loop over 255 is slow.
    # Let's check if we can vectorize or use numpy.
    # Numpy is available.
    
    # Let's try a hybrid: Use the fast delta logic in Python but optimize the loop.
    # Or, stick to the bitmask but use a precomputed list of d's that are "potentially affected".
    # Actually, the bitmask check `if (mask & (mask >> d) & (mask >> (2 * d))) != 0` is very fast in Python if we just do it for all d.
    # 255 iterations of bitwise ops on big integers (512 bits) is likely ~10-50 microseconds.
    # 1000 evaluations = 50ms. 
    # The bottleneck in the seed was probably the overhead of creating new lists/masks or the local search depth.
    
    # Let's implement the `cnt_d` approach. It is more robust for local search.
    
    class ScoreTracker:
        def __init__(self, max_val=512):
            self.max_val = max_val
            self.max_d = max_val // 2  # Max d for 3-AP in 1..512 is 255 (1, 256, 511)
            self.in_set = [False] * (max_val + 1)
            self.cnt_d = [0] * (self.max_d + 1)
            self.score = 0
            
        def reset(self, elements):
            self.in_set = [False] * (self.max_val + 1)
            self.cnt_d = [0] * (self.max_d + 1)
            self.score = 0
            for x in elements:
                self.add(x)
                
        def add(self, x):
            if self.in_set[x]:
                return
            self.in_set[x] = True
            for d in range(1, self.max_d + 1):
                # Check 1: x is the end (x-2d, x-d, x)
                if x - 2 * d >= 1 and self.in_set[x - 2 * d] and self.in_set[x - d]:
                    if self.cnt_d[d] == 0:
                        self.score += 1
                    self.cnt_d[d] += 1
                # Check 2: x is the start (x, x+d, x+2d)
                if x + 2 * d <= self.max_val and self.in_set[x + d] and self.in_set[x + 2 * d]:
                    if self.cnt_d[d] == 0:
                        self.score += 1
                    self.cnt_d[d] += 1
                # Check 3: x is the middle (x-d, x, x+d)
                if x - d >= 1 and x + d <= self.max_val and self.in_set[x - d] and self.in_set[x + d]:
                    if self.cnt_d[d] == 0:
                        self.score += 1
                    self.cnt_d[d] += 1
            self.in_set[x] = True # Ensure it's set, though we set it earlier
            
        def remove(self, x):
            if not self.in_set[x]:
                return
            self.in_set[x] = False
            for d in range(1, self.max_d + 1):
                if x - 2 * d >= 1 and self.in_set[x - 2 * d] and self.in_set[x - d]:
                    self.cnt_d[d] -= 1
                    if self.cnt_d[d] == 0:
                        self.score -= 1
                if x + 2 * d <= self.max_val and self.in_set[x + d] and self.in_set[x + 2 * d]:
                    self.cnt_d[d] -= 1
                    if self.cnt_d[d] == 0:
                        self.score -= 1
                if x - d >= 1 and x + d <= self.max_val and self.in_set[x - d] and self.in_set[x + d]:
                    self.cnt_d[d] -= 1
                    if self.cnt_d[d] == 0:
                        self.score -= 1
                        
        def get_score(self):
            return self.score

    # Local Search using ScoreTracker
    def local_search(elements, tracker, max_passes=5):
        """
        Exhaustive single-move local search.
        Tries to replace each element with any other value in 1..512.
        """
        current_elements = elements[:]
        current_score = tracker.get_score()
        
        for _ in range(max_passes):
            improved = False
            for i in range(N):
                old_val = current_elements[i]
                tracker.remove(old_val)
                
                best_new_val = old_val
                best_score = current_score
                
                # Try all possible new values
                # Optimization: We can precompute valid values? No, just loop 1..512.
                for new_val in range(1, MAX_VAL + 1):
                    if new_val == old_val:
                        continue
                    if new_val in current_elements:
                        # This check is O(N) if we use a list. 
                        # We should use a set or the tracker's in_set.
                        # tracker.in_set is available.
                        if tracker.in_set[new_val]:
                            continue
                    
                    tracker.add(new_val)
                    s = tracker.get_score()
                    if s > best_score:
                        best_score = s
                        best_new_val = new_val
                    tracker.remove(new_val)
                
                if best_new_val != old_val:
                    # Apply best move
                    current_elements[i] = best_new_val
                    tracker.add(best_new_val)
                    current_score = best_score
                    improved = True
                else:
                    # Re-add old val
                    tracker.add(old_val)
            
            if not improved:
                break
        return current_elements, current_score

    def swap_local_search(elements, tracker, max_passes=2):
        """
        Local search that swaps two elements.
        """
        current_elements = elements[:]
        current_score = tracker.get_score()
        
        for _ in range(max_passes):
            improved = False
            for i in range(N):
                for j in range(i + 1, N):
                    v_i = current_elements[i]
                    v_j = current_elements[j]
                    if v_i == v_j: continue
                    
                    # Remove both
                    tracker.remove(v_i)
                    tracker.remove(v_j)
                    
                    # Try swapping with all other pairs? No, that's O(N^2 * 512^2) which is too slow.
                    # Just try swapping v_i with a random set of values, and v_j with a random set?
                    # Or, try replacing v_i with any value, and v_j with any value?
                    # That's effectively two single moves.
                    
                    # Let's stick to single moves for speed, but add a "random restart" mechanism.
                    tracker.add(v_i)
                    tracker.add(v_j)
            # If no improvement in single moves, try a random perturbation
            if not improved:
                # Pick 2 random elements and swap them with 2 random other values
                if random.random() < 0.5:
                    idxs = random.sample(range(N), 2)
                    i, j = idxs
                    v_i = current_elements[i]
                    v_j = current_elements[j]
                    
                    tracker.remove(v_i)
                    tracker.remove(v_j)
                    
                    new_v_i = random.randint(1, MAX_VAL)
                    new_v_j = random.randint(1, MAX_VAL)
                    
                    # Ensure distinctness
                    attempts = 0
                    while (new_v_i in current_elements or new_v_j in current_elements or new_v_i == new_v_j) and attempts < 10:
                        new_v_i = random.randint(1, MAX_VAL)
                        new_v_j = random.randint(1, MAX_VAL)
                        attempts += 1
                    if new_v_i not in current_elements and new_v_j not in current_elements and new_v_i != new_v_j:
                         tracker.add(new_v_i)
                         tracker.add(new_v_j)
                         current_elements[i] = new_v_i
                         current_elements[j] = new_v_j
                         improved = True # We accept this move to escape local min, even if score drops?
                         # For now, let's only accept if score improves or is equal, to keep it simple.
                         # Actually, for SA, we would accept with probability. 
                         # Let's just do greedy for now.
                         if tracker.get_score() < current_score:
                             # Revert
                             tracker.remove(new_v_i)
                             tracker.remove(new_v_j)
                             tracker.add(v_i)
                             tracker.add(v_j)
                             current_elements[i] = v_i
                             current_elements[j] = v_j
                             improved = False
                         else:
                             current_score = tracker.get_score()
                    else:
                         tracker.add(v_i)
                         tracker.add(v_j)
                         
        return current_elements, current_score

    # Initialize
    tracker = ScoreTracker(MAX_VAL)
    
    best_elements = None
    best_score = -1
    
    # Multiple restarts
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Generate initial population
        if r == 0:
            # Start with a good structured guess: Interval
            elements = list(range(1, N + 1))
        else:
            # Random start
            elements = random.sample(range(1, MAX_VAL + 1), N)
            
        tracker.reset(elements)
        current_score = tracker.get_score()
        
        # Local Search
        elements, current_score = local_search(elements, tracker, max_passes=3)
        
        # Swap Local Search (lighter)
        elements, current_score = swap_local_search(elements, tracker, max_passes=1)
        
        if current_score > best_score:
            best_score = current_score
            best_elements = elements[:]
            print(f"Restart {r}: Score {current_score}", file=sys.stderr)
        
        # If we have time, do more local search on the best found so far
        if time.time() < deadline - 10:
            # Re-reset tracker to best elements
            tracker.reset(best_elements)
            # More aggressive local search
            best_elements, best_score = local_search(best_elements, tracker, max_passes=2)
            if best_score > best_score: # Redundant check, but safe
                 pass

    # Final polish
    if best_elements is not None and time.time() < deadline - 2:
        tracker.reset(best_elements)
        # One more exhaustive pass
        best_elements, best_score = local_search(best_elements, tracker, max_passes=1)
        
    return best_elements

# EVOLVE-BLOCK-END
