# ... inside construct
# Generate all 19 cells of R=2
all_cells = []
for q in range(-2, 3):
    for r in range(-2, 3):
        if abs(q) <= 2 and abs(r) <= 2 and abs(q+r) <= 2:
            all_cells.append((q, r))
# Sort by distance from center
# Pick 11 closest?
# Or pick 11 that fit in Side 4?
