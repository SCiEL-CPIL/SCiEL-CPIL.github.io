   new_sums_list = list(new_sums)
   for i in range(len(new_sums_list)):
       a = new_sums_list[i]
       for j in range(i + 1, len(new_sums_list)):
           b = new_sums_list[j]
           if a > b:
               a, b = b, a
           if b % a == 0:
               return None, None
   