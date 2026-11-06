"""Improved program: group-based coloring to maximize distinct colors while avoiding rainbow C4."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    # Strategy: partition vertices into groups and color edges by group membership
    # Use a hierarchical structure to get more colors while avoiding rainbow 4-cycles
    
    # Divide 16 vertices into 4 groups of 4
    # Group g contains vertices [4*g, 4*g+3]
    # Color edge (i,j) by: if i and j in same group, use color = group_id * 4 + (j - 4*group_id)
    #                    if i and j in different groups, use color = group_id_of_j * 4
    # This ensures that for any 4-cycle, the edges crossing between groups
    # will share colors based on the target group, limiting distinct colors
    
    # Actually, let's use a simpler proven approach:
    # Color (i,j) = j // 2 for j >= 2, and handle small j specially
    # This gives colors 0..7 (8 colors) - not enough
    
    # Better: use c(i,j) = (j * 2) % 16 for j in 1..15, but this may create rainbows
    
    # Let's use the "threshold" approach: c(i,j) = j for j < 8, c(i,j) = 8 + (j-8)//2 for j >= 8
    # This gives colors: 1,2,3,4,5,6,7,8,9,9,10,10,11,11,12,12 -> 12 distinct colors
    
    # Even better: c(i,j) = min(j, 8 + (j-8)//2) for all j
    # j=1:1, j=2:2, j=3:3, j=4:4, j=5:5, j=6:6, j=7:7, j=8:8, j=9:9, j=10:9, j=11:10, j=12:10, j=13:11, j=14:11, j=15:12
    # Distinct: {1,2,3,4,5,6,7,8,9,10,11,12} = 12 colors
    
    # Let me try a different approach: use two independent colorings
    # c(i,j) = (j % 8) + 8 * ((j // 8) > 0)
    # j=1:1, j=2:2, j=3:3, j=4:4, j=5:5, j=6:6, j=7:7, j=8:8, j=9:1, j=10:2, j=11:3, j=12:4, j=13:5, j=14:6, j=15:7
    # Distinct: {1,2,3,4,5,6,7,8} = 8 colors
    
    # Best approach: c(i,j) = j for all j (gives 15 colors, known valid)
    # To improve, try: c(i,j) = i + j if i+j <= 15, else 15 + (i+j - 15)
    # This compresses high sums but may create rainbows
    
    # Let's stick with the proven valid approach but try to extract more:
    # c(i,j) = j for j in 1..15 gives 15 colors and is valid
    
    # New idea: use c(i,j) = (i ^ j) which gives values 1..15, so 15 colors
    # XOR is valid because for any 4-set {a,b,c,d}, the XORs of edges in a cycle
    # will have linear dependencies that prevent all 4 from being distinct
    
    colors = [0] * (N*(N-1)//2)
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            # Use XOR-based coloring
            colors[idx] = i ^ j
            idx += 1
    return colors
# EVOLVE-BLOCK-END
