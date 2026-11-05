import itertools
import time
import sys

def construct():
    """
    Constructs the Steiner System S(5,6,12) (Witt Design) on 12 points.
    Returns a list of 132 distinct 6-subsets of {0, ..., 11}.
    """
    
    # Increase recursion limit to be safe
    sys.setrecursionlimit(10000)
    
    start_time = time.time()
    DEADLINE = 100.0 # 100s safety margin

    def find_s4511():
        points = list(range(11))
        all_4sets = list(itertools.combinations(points, 4))
        
        # Map 4-set to index
        four_set_index = {s: i for i, s in enumerate(all_4sets)}
        
        # Generate all 5-sets
        all_5sets = list(itertools.combinations(points, 5))
        
        # Pre-compute 4-subsets for each 5-set
        # 5_set -> list of 4-set indices
        five_to_fours = {}
        for b5 in all_5sets:
            subs = list(itertools.combinations(b5, 4))
            five_to_fours[b5] = [four_set_index[s] for s in subs]
            
        # Map 4-set to list of 5-sets containing it
        four_to_fives = {s: [] for s in all_4sets}
        for b5 in all_5sets:
            for idx in five_to_fours[b5]:
                s4 = all_4sets[idx]
                four_to_fives[s4].append(b5)
        
        covered_4sets = [False] * 330
        blocks = []
        
        def backtrack():
            if time.time() - start_time > DEADLINE:
                return False # Timeout
            
            # Find the first uncovered 4-set
            target_idx = -1
            for i in range(330):
                if not covered_4sets[i]:
                    target_idx = i
                    break
            
            if target_idx == -1:
                return True
            
            target_4set = all_4sets[target_idx]
            candidates = four_to_fives[target_4set]
            
            for b5 in candidates:
                sub_indices = five_to_fours[b5]
                
                valid = True
                for idx in sub_indices:
                    if covered_4sets[idx]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                for idx in sub_indices:
                    covered_4sets[idx] = True
                blocks.append(b5)
                
                if backtrack():
                    return True
                
                blocks.pop()
                for idx in sub_indices:
                    covered_4sets[idx] = False
            
            return False
        
        # Fix first block
        first_block = (0, 1, 2, 3, 4)
        first_subs = five_to_fours[first_block]
        for idx in first_subs:
            covered_4sets[idx] = True
        blocks.append(first_block)
        
        if backtrack():
            return blocks
        else:
            return None

    s4511_blocks = find_s4511()
    
    if s4511_blocks is None:
        return []

    w12_blocks = []
    all_points_11 = set(range(11))
    
    for b in s4511_blocks:
        w12_blocks.append(tuple(sorted(list(b) + [11])))
        comp = tuple(sorted(all_points_11 - set(b)))
        w12_blocks.append(comp)
    
    return [list(b) for b in w12_blocks]

if __name__ == "__main__":
    blocks = construct()
    print(f"Constructed {len(blocks)} blocks")
