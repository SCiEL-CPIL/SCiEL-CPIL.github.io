"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # The proven valid upper tail starts at 171
    # For a in [171, 256], 3*a >= 513 > max(b+c)=512
    # Thus a cannot divide b+c for any b,c in the set with a < b <= c
    best_set = list(range(171, 257))
    
    # Optional: Try to see if we can squeeze in one more element by pruning
    # This is a heuristic search for improvement
    # Given the analysis, adding x < 171 requires removing >= 2 elements usually
    # So we stick to the tail unless we find a specific case where 1 removal fixes it
    
    # Check if 170 can be added with 1 removal
    # Conflicts for 170: sums = 510. Pairs (254,256), (255,255).
    # Removing 255 breaks (255,255) but not (254,256).
    # Removing 254 breaks (254,256) but not (255,255).
    # Need 2 removals. Net -1.
    
    # Check if 169 can be added
    # Conflicts for 169: sums = 507. Pairs (251,256), (252,255), (253,254).
    # 3 disjoint pairs. Need 3 removals. Net -2.
    
    # It seems the tail [171, 256] is locally optimal against single additions.
    
    return best_set
# EVOLVE-BLOCK-END
