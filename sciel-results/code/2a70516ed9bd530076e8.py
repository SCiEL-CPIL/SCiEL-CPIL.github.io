"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 17
    
    # Subgroups of Z_17^* of order 4 and their cosets
    # H = {1, 4, 13, 16}
    # 2H = {2, 8, 9, 15}
    # 3H = {3, 5, 12, 14}
    # 6H = {6, 7, 10, 11}
    
    sets = [
        [1, 4, 13, 16],
        [2, 8, 9, 15],
        [3, 5, 12, 14],
        [6, 7, 10, 11]
    ]
    
    # Map difference to color index
    diff_to_color = {}
    for k, s in enumerate(sets):
        for d in s:
            diff_to_color[d] = k
            
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            # d is in 1..16
            colors.append(diff_to_color[d])
            
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
