"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Valid 5-coloring of K_26 without monochromatic C4.
    # Based on cyclic distance partitioning.
    n = 26
    
    # Partition of distances 1..12 (and 13 is not needed as min(d, n-d) covers it? 
    # No, for n=26, distances are 1..12. 
    # Wait, min(j-i, n-(j-i)) for n=26 gives 1..12.
    # My previous analysis for n=26 used distances 1..12?
    # Let's re-verify.
    # For n=26, max distance is 13.
    # min(j-i, 26-(j-i)).
    # If j-i=13, min(13, 13)=13.
    # So distances are 1..13.
    # I need to partition 1..13.
    
    # Let's re-verify the partition for n=26.
    # S1={1,4}, S2={2,5}, S3={3,7}, S4={6,9,13}, S5={8,10,11,12}
    # Check S4={6,9,13} for n=26.
    # 6+6=12 not in S4.
    # 6+9=15 not in S4.
    # 6+13=19 not in S4.
    # 9+9=18 not in S4.
    # 9+13=22 not in S4.
    # 13+13=26=0 not in S4.
    # Diffs: 9-6=3, 13-6=7, 13-9=4. None in S4.
    # Check S5={8,10,11,12} for n=26.
    # 8+8=16 not in S5.
    # 8+10=18 not in S5.
    # 8+11=19 not in S5.
    # 8+12=20 not in S5.
    # 10+10=20 not in S5.
    # 10+11=21 not in S5.
    # 10+12=22 not in S5.
    # 11+11=22 not in S5.
    # 11+12=23 not in S5.
    # 12+12=24 not in S5.
    # Diffs: 10-8=2, 11-8=3, 12-8=4, 11-10=1, 12-10=2, 12-11=1. None in S5.
    
    # Check other sets for n=26.
    # S1={1,4}. 1+1=2, 1+4=5, 4+4=8. Diffs 3. OK.
    # S2={2,5}. 2+2=4, 2+5=7, 5+5=10. Diffs 3. OK.
    # S3={3,7}. 3+3=6, 3+7=10, 7+7=14. Diffs 4. OK.
    
    # All sets are sum-free and difference-free.
    # Thus the circulant graphs are C4-free.
    
    color_map = {
        0: {1, 4},
        1: {2, 5},
        2: {3, 7},
        3: {6, 9, 13},
        4: {8, 10, 11, 12}
    }
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            dist = min(d, n - d)
            # Find color
            for c in range(5):
                if dist in color_map[c]:
                    colors.append(c)
                    break
            else:
                # Should not happen
                colors.append(0)
                
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
