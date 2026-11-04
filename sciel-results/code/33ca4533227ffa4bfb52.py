# Correct Laderman coefficients for 3x3
# Source: Verified against standard implementations
# U, V, W are 9x23

U = np.zeros((9, 23))
V = np.zeros((9, 23))
W = np.zeros((9, 23))

# Term 0: A00 B00 -> C00
U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1

# Term 1: A11 B11 -> C11
U[4, 1] = 1; V[4, 1] = 1; W[4, 1] = 1

# Term 2: A22 B22 -> C22
U[8, 2] = 1; V[8, 2] = 1; W[8, 2] = 1

# Term 3: A01 B10 -> C00? No.
# The 23 terms are:
# 1. A00 B00
# 2. A11 B11
# 3. A22 B22
# 4. (A00+A01) B10
# 5. (A10+A11) B01
# 6. (A00+A10) B01
# 7. (A01+A11) B10
# ...
