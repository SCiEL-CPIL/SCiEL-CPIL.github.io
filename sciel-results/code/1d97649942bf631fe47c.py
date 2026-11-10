   def can_add(y, S_set):
       # y is largest or smallest
       for b in S_set:
           a = 2 * b - y
           if a in S_set:
               return False
       # y is middle
       for a in S_set:
           c = 2 * y - a
           if c in S_set:
               return False
       return True
   