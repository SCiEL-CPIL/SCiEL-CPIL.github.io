# Laderman's 23 terms for 3x3 matrix multiplication
# a is 3x3, b is 3x3
# c = a @ b
# The 23 products are:
# 1. (a00+a01+a02)*(b00+b01+b02)
# 2. (a00+a01+a02)*(b00+b11+b22)
# 3. (a00+a01+a02)*(b00+b22+b11) # Wait, b11+b22 is same as b22+b11
# Actually, Laderman's algorithm is:
# M1 = (a11+a12+a13)(b11+b12+b13)
# M2 = (a11+a12+a13)(b11+b22+b33)
# M3 = (a11+a12+a13)(b11+b33+b22)
# ...
