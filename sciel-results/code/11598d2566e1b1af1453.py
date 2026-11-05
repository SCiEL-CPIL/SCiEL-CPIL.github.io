   def transversals(pairs):
       (a,b), (c,d), (e,f) = pairs
       for x in (a,b):
           for y in (c,d):
               for z in (e,f):
                   yield tuple(sorted((x,y,z)))
   