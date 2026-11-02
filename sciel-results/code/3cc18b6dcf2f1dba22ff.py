# For 2x3x3, let's try to find rank-11
# U: (6, 11), V: (9, 11), W: (6, 11)
# 66 unknowns in U, V, W combined... no, 6*11 + 9*11 + 6*11 = 66+99+66 = 231 unknowns
# 18 equations
# This is underdetermined, so solutions should exist if rank >= 11

# Let me fix W and solve for U, V
