   def _decode(code, coeffs, offset):
       temp = int(code)
       for k in range(15):
           # Extract digit
           # In balanced ternary, digit is -1, 0, 1
           # We can use:
           digit = temp % 3
           # But this gives 0, 1, 2.
           # If digit == 2, it means -1?
           # Let's check:
           # code = -1.
           # -1 % 3 = 2.
           # -1 // 3 = -1.
           # This doesn't work.
           
           # Correct way:
           # r = temp % 3
           # if r == 2: digit = -1, temp = (temp + 1) // 3
           # elif r == 1: digit = 1, temp = (temp - 1) // 3
           # else: digit = 0, temp = temp // 3
           
           r = temp % 3
           if r == 2:
               coeffs[offset + k] = -1
               temp = (temp + 1) // 3
           elif r == 1:
               coeffs[offset + k] = 1
               temp = (temp - 1) // 3
           else:
               coeffs[offset + k] = 0
               temp = temp // 3
   