import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def get_triples(n):
        return list(itertools.combinations(range(n), 3))
    
    def get_k4s(n):
        return list(itertools.combinations(range(n), 4))
        
    def get_k5s(n):
        return list(itertools.combinations(range(n), 5))
        
    def count_violations(colors, n, k4s, k5s, triple_indices):
        # triple_indices: map from triple (a,b,c) to index in colors
        # Pre-compute which k4s and k5s contain each triple? 
        # For speed, we should maintain counts incrementally.
        pass

    # We will use a class-like structure for the search to manage state
    class HypergraphSearch:
        def __init__(self, n):
            self.n = n
            self.triples = list(itertools.combinations(range(n), 3))
            self.num_triples = len(self.triples)
            self.triple_index = {t: i for i, t in enumerate(self.triples)}
            
            self.k4s = list(itertools.combinations(range(n), 4))
            self.k5s = list(itertools.combinations(range(n), 5))
            
            # Map each triple to the list of k4s and k5s it belongs to
            self.triple_to_k4s = [[] for _ in range(self.num_triples)]
            self.triple_to_k5s = [[] for _ in range(self.num_triples)]
            
            # k4s are 4-sets. Each k4 contains 4 triples.
            for idx, k4 in enumerate(self.k4s):
                for t in itertools.combinations(k4, 3):
                    ti = self.triple_index[t]
                    self.triple_to_k4s[ti].append(idx)
                    
            # k5s are 5-sets. Each k5 contains 10 triples.
            for idx, k5 in enumerate(self.k5s):
                for t in itertools.combinations(k5, 3):
                    ti = self.triple_index[t]
                    self.triple_to_k5s[ti].append(idx)
                    
            # State: colors array of 0s and 1s
            self.colors = [random.randint(0, 1) for _ in range(self.num_triples)]
            
            # Counts of red triples in each k4, blue triples in each k5
            self.red_counts_4 = [0] * len(self.k4s)
            self.blue_counts_5 = [0] * len(self.k5s)
            
            self.total_violations = 0
            self._update_counts()

        def _update_counts(self):
            # Initialize counts from scratch
            for i, c in enumerate(self.colors):
                if c == 0: # Red
                    for k4_idx in self.triple_to_k4s[i]:
                        self.red_counts_4[k4_idx] += 1
                else: # Blue
                    for k5_idx in self.triple_to_k5s[i]:
                        self.blue_counts_5[k5_idx] += 1
            
            self.total_violations = 0
            for count in self.red_counts_4:
                if count == 4:
                    self.total_violations += 1
            for count in self.blue_counts_5:
                if count == 10:
                    self.total_violations += 1

        def flip_triple(self, ti, new_color):
            old_color = self.colors[ti]
            if old_color == new_color:
                return 0
            
            delta_violations = 0
            
            if old_color == 0: # Was Red, becoming Blue
                # Remove from red counts of k4s
                for k4_idx in self.triple_to_k4s[ti]:
                    old_count = self.red_counts_4[k4_idx]
                    self.red_counts_4[k4_idx] = old_count - 1
                    if old_count == 4:
                        delta_violations -= 1
                    elif old_count - 1 == 4:
                        delta_violations += 1
                
                # Add to blue counts of k5s
                for k5_idx in self.triple_to_k5s[ti]:
                    old_count = self.blue_counts_5[k5_idx]
                    self.blue_counts_5[k5_idx] = old_count + 1
                    if old_count == 10:
                        delta_violations -= 1
                    elif old_count + 1 == 10:
                        delta_violations += 1
                        
            else: # Was Blue, becoming Red
                # Remove from blue counts of k5s
                for k5_idx in self.triple_to_k5s[ti]:
                    old_count = self.blue_counts_5[k5_idx]
                    self.blue_counts_5[k5_idx] = old_count - 1
                    if old_count == 10:
                        delta_violations -= 1
                    elif old_count - 1 == 10:
                        delta_violations += 1
                
                # Add to red counts of k4s
                for k4_idx in self.triple_to_k4s[ti]:
                    old_count = self.red_counts_4[k4_idx]
                    self.red_counts_4[k4_idx] = old_count + 1
                    if old_count == 4:
                        delta_violations -= 1
                    elif old_count + 1 == 4:
                        delta_violations += 1

            self.colors[ti] = new_color
            self.total_violations += delta_violations
            return delta_violations

        def search(self, time_limit):
            end_time = time.time() + time_limit
            T = 1.0
            T_min = 0.01
            cooling = 0.999
            best_violations = self.total_violations
            best_colors = self.colors[:]
            
            while time.time() < end_time:
                ti = random.randint(0, self.num_triples - 1)
                new_color = 1 - self.colors[ti]
                
                # Calculate delta
                delta = self.flip_triple(ti, new_color)
                
                if delta < 0 or random.random() < pow(2.71828, -delta / T):
                    if self.total_violations < best_violations:
                        best_violations = self.total_violations
                        best_colors = self.colors[:]
                        if best_violations == 0:
                            break
                else:
                    self.flip_triple(ti, self.colors[ti]) # Revert
                
                T *= cooling
                if T < T_min:
                    T = T_min
            
            self.colors = best_colors
            self._update_counts() # Ensure consistency
            return self.total_violations, self.colors

    best_result = None
    
    # Try n from 34 down to 10
    # Allocate time based on n? Larger n is harder. 
    # Let's give more time to smaller n if larger n fails quickly?
    # Or just iterate and break if found.
    
    # Strategy: Try 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20
    # Time budget: 110s total.
    # If we fail 34, we have 110s for 33? No, we should share time.
    # Let's try to find a solution for a specific n.
    # If we find one, return it.
    
    # To be safe, let's start with a known small n to ensure we have a fallback, 
    # but we want to maximize.
    # Let's try n=20 first with a short timeout to see if it's easy, then go higher?
    # No, n=20 is easy. n=34 is hard.
    # Let's try n=34 with 10s, n=33 with 10s, etc.
    
    ns_to_try = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]
    
    time_per_n = 4.0 # seconds
    
    for n in ns_to_try:
        if time.time() > deadline - 5:
            break
            
        search = HypergraphSearch(n)
        # Run search for time_per_n
        violations, colors = search.search(time_per_n)
        
        if violations == 0:
            return {'n': n, 'colors': colors}
            
    # If we didn't find any, return the seed n=9
    # But wait, n=9 is very small. We should probably find something better.
    # The loop above tries down to 10. 
    # If all fail, we return n=9 seed.
    
    # Actually, we can just return the seed if all fail.
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]}
